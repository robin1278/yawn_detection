"""
Yawn Detection State Machine and Progression Tracker.

Implements the exact 4-stage lifecycle:
1. Mouth Opening Increases
2. Mouth Fully Opening (with sustained peak hold)
3. Mouth Closing
4. Mouth Closing More (return to resting closed)
All within an anatomically validated duration window (2.5s to 10.5s).
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from config import DetectorConfig


class YawnStage(Enum):
    IDLE = 0
    OPENING_INCREASES = 1
    FULLY_OPENING = 2
    MOUTH_CLOSING = 3
    CLOSING_MORE = 4

    # Backward compatibility aliases
    MOUTH_OPENING = 1
    MOUTH_OPENING_MORE = 1
    MOUTH_FINALLY_OPENED = 2
    MOUTH_FINALLY_CLOSED = 4

    @property
    def display_name(self) -> str:
        names = {
            YawnStage.IDLE: "Mouth Closed (Resting)",
            YawnStage.OPENING_INCREASES: "Opening Increases",
            YawnStage.FULLY_OPENING: "Fully Opening (Peak)",
            YawnStage.MOUTH_CLOSING: "Mouth Closing",
            YawnStage.CLOSING_MORE: "Closing More",
        }
        return names.get(self, "Unknown")

    @property
    def short_name(self) -> str:
        short = {
            YawnStage.IDLE: "RESTING",
            YawnStage.OPENING_INCREASES: "1. OPENING INCREASES",
            YawnStage.FULLY_OPENING: "2. FULLY OPENING",
            YawnStage.MOUTH_CLOSING: "3. MOUTH CLOSING",
            YawnStage.CLOSING_MORE: "4. CLOSING MORE",
        }
        return short.get(self, "")


@dataclass
class YawnEvent:
    timestamp: float
    duration: float
    peak_mar: float
    stage_durations: Dict[str, float]
    formatted_time: str = ""

    def __post_init__(self):
        if not self.formatted_time:
            self.formatted_time = time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class YawnDetector:
    """
    Finite State Machine for detecting yawn sequences meeting
    the 4-phase criteria (Opening Increases -> Fully Opening -> Closing -> Closing More).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

        # Current state machine status
        self.current_stage: YawnStage = YawnStage.IDLE
        self.cycle_start_time: float = 0.0
        self.stage_start_times: Dict[YawnStage, float] = {}
        self.stage_durations: Dict[str, float] = {}
        self.peak_mar: float = 0.0

        # Statistics & History
        self.yawn_count: int = 0
        self.recent_events: List[YawnEvent] = []
        self.last_status_message: str = "Ready. Monitoring facial landmarks..."
        self.status_message_time: float = time.time()
        self.alert_active: bool = False
        self.alert_start_time: float = 0.0
        self.last_yawn_duration: float = 0.0

        # Smoothed MAR value
        self.smoothed_mar: float = 0.0
        self.raw_mar: float = 0.0

        # Dynamic calibration offset
        self.baseline_mar: float = 0.12
        self.baseline_lip_chin_ratio: float = 0.60
        self.is_calibrating: bool = False
        self.calibration_samples: List[float] = []
        self.calibration_chin_samples: List[float] = []
        self.last_resting_time: float = 0.0
        self.last_update_time: float = 0.0
        self.closing_confirm_count: int = 0

        # Peak hold & Monotonic closing tracking
        self.peak_hold_start: float = 0.0
        self.min_closing_mar: float = 1.0
        self.is_completed_hold: bool = False
        self.completed_display_time: float = 0.0
        self.has_sustained_peak: bool = False

    def calibrate_baseline(self, mar: float, landmark_data: Optional[Dict[str, Any]] = None):
        """Accumulate samples to compute natural resting mouth aspect ratio and facial proportions."""
        self.calibration_samples.append(mar)
        if landmark_data and "jaw_ratio" in landmark_data:
            self.calibration_chin_samples.append(landmark_data["jaw_ratio"])

        if len(self.calibration_samples) >= 60:  # ~2 seconds at 30fps
            avg_baseline = sum(self.calibration_samples) / len(self.calibration_samples)
            # Enforce safe baseline floor of 0.12 to prevent false speech/smile triggers
            self.baseline_mar = max(0.12, min(avg_baseline, 0.22))
            if self.calibration_chin_samples:
                self.baseline_lip_chin_ratio = float(sum(self.calibration_chin_samples) / len(self.calibration_chin_samples))

            # Adapt thresholds relative to baseline with strict minimum bounds
            self.config.mar_thresh_closed = max(0.16, self.baseline_mar + 0.06)
            self.config.mar_thresh_opening = max(0.30, self.baseline_mar + 0.16)
            self.config.mar_thresh_mid = max(0.44, self.baseline_mar + 0.30)
            self.config.mar_thresh_finally_opened = max(0.65, self.baseline_mar + 0.48)
            self.is_calibrating = False
            self.calibration_samples.clear()
            self.calibration_chin_samples.clear()
            self._set_status(f"Calibrated! Baseline MAR: {self.baseline_mar:.2f} | Chin Ratio: {self.baseline_lip_chin_ratio:.2f}")

    def start_calibration(self):
        """Begin calibration routine."""
        self.is_calibrating = True
        self.calibration_samples.clear()
        self.calibration_chin_samples.clear()
        self._set_status("Calibrating resting mouth... Keep mouth relaxed and closed.")

    def reset_stats(self):
        """Reset counters and history."""
        self.yawn_count = 0
        self.recent_events.clear()
        self._reset_to_idle("Yawn statistics reset.")

    def _set_status(self, message: str):
        self.last_status_message = message
        self.status_message_time = time.time()

    def _reset_to_idle(self, reason: Optional[str] = None):
        """Reset state machine back to IDLE."""
        if reason:
            self._set_status(reason)
        self.current_stage = YawnStage.IDLE
        self.cycle_start_time = 0.0
        self.stage_start_times.clear()
        self.stage_durations.clear()
        self.peak_mar = 0.0
        self.last_resting_time = 0.0
        self.closing_confirm_count = 0
        self.peak_hold_start = 0.0
        self.min_closing_mar = 1.0
        self.is_completed_hold = False
        self.completed_display_time = 0.0
        self.has_sustained_peak = False

    def get_elapsed_time(self, current_time: Optional[float] = None) -> float:
        """Returns elapsed time in current cycle, or 0.0 if IDLE."""
        if self.current_stage == YawnStage.IDLE or self.cycle_start_time == 0.0:
            return 0.0
        if self.is_completed_hold and self.last_yawn_duration > 0:
            return self.last_yawn_duration
        now = self.last_update_time if current_time is None else current_time
        return max(0.0, now - self.cycle_start_time)

    def update(self, raw_mar: float, current_time: Optional[float] = None, landmark_data: Optional[Dict[str, Any]] = None) -> Tuple[YawnStage, bool]:
        """
        Update the detector with the latest MAR measurement.
        Returns:
            (current_stage, is_new_yawn_detected)
        """
        now = time.time() if current_time is None else current_time
        self.last_update_time = now
        self.raw_mar = raw_mar

        # Smooth the MAR value using Exponential Moving Average
        if self.smoothed_mar == 0.0:
            self.smoothed_mar = raw_mar
        else:
            alpha = self.config.mar_smoothing_factor
            self.smoothed_mar = (alpha * raw_mar) + ((1.0 - alpha) * self.smoothed_mar)

        mar = self.smoothed_mar

        # Handle calibration if active
        if self.is_calibrating:
            self.calibrate_baseline(mar, landmark_data=landmark_data)
            return self.current_stage, False

        # Clear alert status after 3.0 seconds
        if self.alert_active and (now - self.alert_start_time > 3.0):
            self.alert_active = False

        new_yawn_confirmed = False
        t_closed = max(self.baseline_mar + 0.06, getattr(self.config, 'mar_thresh_closed', 0.18))
        t_open = self.config.mar_thresh_opening
        t_mid = getattr(self.config, 'mar_thresh_mid', 0.45)
        t_peak = self.config.mar_thresh_finally_opened
        
        tol = getattr(self.config, 'duration_tolerance', 0.30)
        min_dur = self.config.min_yawn_duration - tol
        max_dur = self.config.max_yawn_duration + tol

        elapsed_cycle = self.get_elapsed_time(now)

        # Global timeout check: Any cycle exceeding max duration (10.5s + tolerance) is reset
        if self.current_stage != YawnStage.IDLE and not self.is_completed_hold and elapsed_cycle > max_dur:
            self._reset_to_idle(f"Cycle exceeded max window ({elapsed_cycle:.1f}s > {self.config.max_yawn_duration:.1f}s) - Timed Out")
            return self.current_stage, False

        # -------------------------------------------------------------
        # STATE MACHINE TRANSITIONS (4 STAGES)
        # -------------------------------------------------------------
        if self.current_stage == YawnStage.IDLE:
            # Keep track of when mouth was last in resting closed state
            if mar <= (t_closed + 0.03):
                self.last_resting_time = now

            # Stage 1 Trigger: Mouth begins opening from resting closed baseline
            if mar >= t_open:
                # A yawn must start from a resting closed mouth (within last 1.5s)
                if self.last_resting_time == 0.0 or (now - self.last_resting_time <= 1.5):
                    self.current_stage = YawnStage.OPENING_INCREASES
                    self.cycle_start_time = self.last_resting_time if self.last_resting_time > 0 else now
                    self.stage_start_times[YawnStage.OPENING_INCREASES] = self.cycle_start_time
                    self.peak_mar = mar
                    self._set_status("Stage 1/4: Mouth opening increases...")

        elif self.current_stage == YawnStage.OPENING_INCREASES:
            self.peak_mar = max(self.peak_mar, mar)
            stage_1_dur = now - self.stage_start_times.get(YawnStage.OPENING_INCREASES, now)

            # If drops back to resting without progressing within 1.0s
            if mar < (t_open - 0.04) and (stage_1_dur > 0.8):
                self._reset_to_idle("Mouth closed before opening further (Brief movement)")
            # Progression to Stage 2 (Fully Opening / Peak)
            elif mar >= t_peak:
                self._record_stage_duration(YawnStage.OPENING_INCREASES, now)
                self.current_stage = YawnStage.FULLY_OPENING
                self.stage_start_times[YawnStage.FULLY_OPENING] = now
                self.peak_hold_start = now
                self._set_status("Stage 2/4: Mouth fully opening (Peak aperture)...")

        elif self.current_stage == YawnStage.FULLY_OPENING:
            self.peak_mar = max(self.peak_mar, mar)
            peak_dur = now - self.peak_hold_start
            min_hold = getattr(self.config, 'min_peak_hold_duration', 0.35)
            if peak_dur >= min_hold:
                self.has_sustained_peak = True

            # Check for transition into closing phase (when mouth drops below wide aperture)
            t_close_entry = getattr(self.config, 'mar_thresh_mid', 0.45) + 0.08
            if mar < t_close_entry:
                # Dwell time discriminator: Speech only momentarily flutters at peak (< 0.35s)
                if not self.has_sustained_peak:
                    self._reset_to_idle(f"Peak not sustained ({peak_dur:.2f}s < {min_hold:.2f}s) - Speech flutter rejected")
                    return self.current_stage, False
                else:
                    self._record_stage_duration(YawnStage.FULLY_OPENING, now)
                    self.current_stage = YawnStage.MOUTH_CLOSING
                    self.stage_start_times[YawnStage.MOUTH_CLOSING] = now
                    self.min_closing_mar = mar
                    self._set_status("Stage 3/4: Mouth closing begins...")
            elif peak_dur > self.config.max_peak_hold_duration:
                self._reset_to_idle(f"Mouth held open too long ({peak_dur:.1f}s) - Timed Out")

        elif self.current_stage == YawnStage.MOUTH_CLOSING:
            self.min_closing_mar = min(self.min_closing_mar, mar)

            # If mouth re-opens to peak aperture, return to peak opening stage (secondary yawn stretch)
            if mar >= t_peak:
                self.current_stage = YawnStage.FULLY_OPENING
                self.stage_start_times[YawnStage.FULLY_OPENING] = now
                self._set_status("Stage 2/4: Mouth fully opening (Secondary stretch)...")
            # Speech filter: If mouth re-opens moderately without reaching peak after starting to close
            elif mar > (self.min_closing_mar + 0.22) and mar >= (t_open + 0.08):
                self._reset_to_idle("Speech detected during closing (Mouth re-opened) - Reset")
                return self.current_stage, False

            # Progression to Stage 4 (Closing More / Near Resting)
            if mar <= t_mid:
                self._record_stage_duration(YawnStage.MOUTH_CLOSING, now)
                self.current_stage = YawnStage.CLOSING_MORE
                self.stage_start_times[YawnStage.CLOSING_MORE] = now
                self.min_closing_mar = mar
                self._set_status("Stage 4/4: Mouth closing more (Returning to resting closed)...")

        elif self.current_stage == YawnStage.CLOSING_MORE:
            # If in post-yawn completion hold (displays green success state for 1.2s)
            if self.is_completed_hold:
                closed_dur = now - self.completed_display_time
                if mar >= t_open:
                    self._reset_to_idle()
                    self.current_stage = YawnStage.OPENING_INCREASES
                    self.cycle_start_time = now
                    self.stage_start_times[YawnStage.OPENING_INCREASES] = now
                    self.peak_mar = mar
                    self._set_status("Stage 1/4: Mouth opening increases...")
                elif closed_dur > 1.2:
                    self._reset_to_idle("Ready. Monitoring facial landmarks...")
                return self.current_stage, False

            self.min_closing_mar = min(self.min_closing_mar, mar)

            # If mouth re-opens back to peak aperture, return to peak opening stage (secondary yawn stretch)
            if mar >= t_peak:
                self.current_stage = YawnStage.FULLY_OPENING
                self.stage_start_times[YawnStage.FULLY_OPENING] = now
                self._set_status("Stage 2/4: Mouth fully opening (Secondary stretch)...")
            # Speech filter: If mouth re-opens during closing more
            elif mar > (self.min_closing_mar + 0.22) and mar >= (t_open + 0.08):
                self._reset_to_idle("Speech detected during closing (Mouth re-opened) - Reset")
                return self.current_stage, False

            # Closure confirmation: true resting mouth baseline or inner lip touch
            inner_gap = landmark_data.get("inner_lip_distance", 999.0) if landmark_data else 999.0
            is_closed_candidate = (mar <= (t_closed + 0.03)) or (inner_gap <= 4.5)

            # Dynamic skeletal veto: Only veto if chin is genuinely dropped far below calibrated baseline
            if landmark_data and landmark_data.get("jaw_ratio", 0.0) > (self.baseline_lip_chin_ratio + 0.22) and inner_gap > 6.0:
                is_closed_candidate = False

            if is_closed_candidate:
                self.closing_confirm_count += 1
            else:
                self.closing_confirm_count = 0

            # Require confirmed closure across consecutive frames
            if self.closing_confirm_count >= 2:
                self.closing_confirm_count = 0
                self._record_stage_duration(YawnStage.CLOSING_MORE, now)

                # Evaluate final duration
                total_duration = now - self.cycle_start_time
                self.last_yawn_duration = total_duration

                if min_dur <= total_duration <= max_dur:
                    # VALID YAWN CONFIRMED!
                    self.yawn_count += 1
                    new_yawn_confirmed = True
                    self.alert_active = True
                    self.alert_start_time = now
                    self.is_completed_hold = True
                    self.completed_display_time = now

                    event = YawnEvent(
                        timestamp=now,
                        duration=total_duration,
                        peak_mar=self.peak_mar,
                        stage_durations=dict(self.stage_durations)
                    )
                    self.recent_events.append(event)
                    if len(self.recent_events) > 10:
                        self.recent_events.pop(0)

                    self._set_status(f"YAWN DETECTED! Duration: {total_duration:.2f}s (Within target window)")
                    self._play_alert_sound()
                elif total_duration < min_dur:
                    self._set_status(f"Rejected: Too fast ({total_duration:.2f}s < {self.config.min_yawn_duration:.1f}s). Not a yawn.")
                    self._reset_to_idle()
                else:
                    self._set_status(f"Rejected: Too slow ({total_duration:.2f}s > {self.config.max_yawn_duration:.1f}s).")
                    self._reset_to_idle()

        return self.current_stage, new_yawn_confirmed

    def _record_stage_duration(self, stage: YawnStage, current_time: float):
        start = self.stage_start_times.get(stage, self.cycle_start_time)
        duration = max(0.0, current_time - start)
        self.stage_durations[stage.name] = duration

    def _play_alert_sound(self):
        """Trigger non-blocking audio notification if enabled."""
        if not self.config.enable_audio_alert:
            return
        try:
            import winsound
            import threading
            def beep():
                try:
                    winsound.Beep(self.config.audio_frequency_hz, self.config.audio_duration_ms)
                except Exception:
                    pass
            threading.Thread(target=beep, daemon=True).start()
        except Exception:
            pass
