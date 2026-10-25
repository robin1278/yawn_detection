"""
High-Tech Cyber/Modern HUD Overlay for Real-Time Yawn Detection.
Renders 6-stage progression, MAR gauge, duration timer, and alert banners.
"""

import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from yawn_tracker import YawnDetector, YawnStage
from config import DetectorConfig


class UIOverlay:
    """
    Renders a visual interface over the camera feed.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

        # Modern palette (BGR)
        self.COLOR_BG = (22, 22, 28)
        self.COLOR_ACCENT = (255, 180, 0)
        self.COLOR_WHITE = (245, 245, 245)
        self.COLOR_MUTED = (140, 140, 150)
        self.COLOR_GREEN = (60, 220, 100)
        self.COLOR_YELLOW = (40, 215, 255)
        self.COLOR_ORANGE = (30, 140, 255)
        self.COLOR_RED = (60, 60, 245)
        self.COLOR_MAGENTA = (220, 80, 255)
        self.COLOR_CYAN = (240, 220, 60)

    def _draw_rounded_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                           color: Tuple[int, int, int], radius: int = 8, thickness: int = -1):
        """Draw a rectangle with rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

        if thickness == -1:
            # Filled rounded rect
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            # Border only
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)

    def _draw_translucent_box(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                              color: Tuple[int, int, int], alpha: float = 0.65, radius: int = 8):
        """Draw an alpha-blended background panel."""
        overlay = img.copy()
        self._draw_rounded_rect(overlay, pt1, pt2, color, radius=radius, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    def get_stage_color(self, stage: YawnStage) -> Tuple[int, int, int]:
        colors = {
            YawnStage.IDLE: (180, 180, 180),
            YawnStage.OPENING_INCREASES: self.COLOR_YELLOW,
            YawnStage.FULLY_OPENING: self.COLOR_MAGENTA,
            YawnStage.MOUTH_CLOSING: self.COLOR_CYAN,
            YawnStage.CLOSING_MORE: self.COLOR_GREEN,
        }
        return colors.get(stage, self.COLOR_WHITE)

    def draw_hud(self, frame: np.ndarray, detector: YawnDetector, fps: float,
                 has_face: bool, landmark_data: Optional[Dict[str, Any]] = None,
                 is_recording: bool = False):
        """
        Draw the complete HUD on the frame.
        """
        h, w, _ = frame.shape
        now = time.time()

        # -------------------------------------------------------------
        # 1. TOP HEADER BAR
        # -------------------------------------------------------------
        self._draw_translucent_box(frame, (10, 10), (w - 10, 68), (15, 15, 20), alpha=0.82, radius=10)
        self._draw_rounded_rect(frame, (10, 10), (w - 10, 68), (50, 50, 65), radius=10, thickness=1)

        # Title
        cv2.putText(frame, "YAWN DETECTION SYSTEM", (28, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, self.COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "MediaPipe FaceMesh (4-Stage Dynamic Arc | 2.5s - 10.5s Window)", (28, 59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Recording indicator
        if is_recording:
            pulse = int((now * 2) % 2)
            rec_color = self.COLOR_RED if pulse == 0 else (100, 100, 240)
            cv2.circle(frame, (w - 325, 38), 6, rec_color, -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "REC", (w - 312, 43),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, rec_color, 1, cv2.LINE_AA)

        # FPS & Audio indicator
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (w - 230, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 1, cv2.LINE_AA)

        audio_text = "SOUND: ON" if self.config.enable_audio_alert else "SOUND: MUTED"
        audio_color = self.COLOR_GREEN if self.config.enable_audio_alert else self.COLOR_MUTED
        cv2.putText(frame, audio_text, (w - 135, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, audio_color, 1, cv2.LINE_AA)

        # Face tracking status pill
        face_color = self.COLOR_GREEN if has_face else self.COLOR_RED
        cv2.circle(frame, (w - 245, 38), 5, face_color, -1, lineType=cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. FOUR STAGE PROGRESSION BAR
        # -------------------------------------------------------------
        stages = [
            (YawnStage.OPENING_INCREASES, "1. OPENING INCREASES"),
            (YawnStage.FULLY_OPENING, "2. FULLY OPENING"),
            (YawnStage.MOUTH_CLOSING, "3. MOUTH CLOSING"),
            (YawnStage.CLOSING_MORE, "4. CLOSING MORE"),
        ]

        bar_y1 = 78
        bar_y2 = 126
        self._draw_translucent_box(frame, (10, bar_y1), (w - 10, bar_y2), (18, 18, 24), alpha=0.85, radius=8)
        self._draw_rounded_rect(frame, (10, bar_y1), (w - 10, bar_y2), (45, 45, 55), radius=8, thickness=1)

        available_w = (w - 20) - 20
        chip_w = (available_w - (len(stages) - 1) * 8) // len(stages)
        current_stage_val = detector.current_stage.value

        for i, (stage, label) in enumerate(stages):
            cx1 = 20 + i * (chip_w + 8)
            cx2 = cx1 + chip_w
            cy1 = bar_y1 + 8
            cy2 = bar_y2 - 8

            is_active = (detector.current_stage == stage)
            is_completed = (current_stage_val > stage.value and current_stage_val != 0)

            if is_active:
                chip_bg = self.get_stage_color(stage)
                text_color = (10, 10, 10)
                # Outer glow effect
                self._draw_rounded_rect(frame, (cx1 - 2, cy1 - 2), (cx2 + 2, cy2 + 2), chip_bg, radius=6, thickness=2)
                self._draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), chip_bg, radius=6, thickness=-1)
            elif is_completed:
                chip_bg = (40, 55, 45)
                text_color = self.COLOR_GREEN
                self._draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), chip_bg, radius=6, thickness=-1)
                self._draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), self.COLOR_GREEN, radius=6, thickness=1)
            else:
                chip_bg = (28, 28, 35)
                text_color = self.COLOR_MUTED
                self._draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), chip_bg, radius=6, thickness=-1)

            # Center text in chip
            font_scale = 0.38 if chip_w < 160 else 0.44
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = cx1 + (chip_w - tw) // 2
            ty = cy1 + (cy2 - cy1 + th) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. LEFT STATS & METRICS PANEL
        # -------------------------------------------------------------
        panel_w = 260
        panel_y1 = 136
        panel_y2 = h - 60
        self._draw_translucent_box(frame, (10, panel_y1), (panel_w, panel_y2), (18, 18, 25), alpha=0.85, radius=10)
        self._draw_rounded_rect(frame, (10, panel_y1), (panel_w, panel_y2), (50, 50, 65), radius=10, thickness=1)

        # Stat: Yawn Counter
        cv2.putText(frame, "TOTAL YAWNS", (25, panel_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{detector.yawn_count}", (25, panel_y1 + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, self.COLOR_WHITE, 2, cv2.LINE_AA)

        # Stat: Current Stage Name
        cv2.putText(frame, "CURRENT PHASE", (25, panel_y1 + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MUTED, 1, cv2.LINE_AA)
        stage_name = detector.current_stage.display_name
        stage_color = self.get_stage_color(detector.current_stage)
        cv2.putText(frame, stage_name, (25, panel_y1 + 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, stage_color, 2, cv2.LINE_AA)

        # Divider line
        cv2.line(frame, (25, panel_y1 + 142), (panel_w - 15, panel_y1 + 142), (45, 45, 55), 1)

        # Stat: MAR Meter (Gauge)
        mar_val = detector.smoothed_mar
        cv2.putText(frame, "MOUTH ASPECT RATIO", (25, panel_y1 + 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"MAR: {mar_val:.3f}", (25, panel_y1 + 195),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, stage_color, 2, cv2.LINE_AA)

        # Horizontal MAR Gauge Bar
        gauge_x1 = 25
        gauge_x2 = panel_w - 20
        gauge_y = panel_y1 + 215
        gauge_w = gauge_x2 - gauge_x1
        cv2.rectangle(frame, (gauge_x1, gauge_y), (gauge_x2, gauge_y + 12), (35, 35, 45), -1)

        # Fill gauge proportional to MAR (0.0 to 1.0)
        fill_pct = min(1.0, max(0.0, mar_val / 1.0))
        fill_w = int(gauge_w * fill_pct)
        if fill_w > 0:
            cv2.rectangle(frame, (gauge_x1, gauge_y), (gauge_x1 + fill_w, gauge_y + 12), stage_color, -1)

        # Threshold Markers on Gauge
        t1_x = int(gauge_x1 + gauge_w * self.config.mar_thresh_opening)
        t2_x = int(gauge_x1 + gauge_w * getattr(self.config, 'mar_thresh_mid', 0.45))
        t3_x = int(gauge_x1 + gauge_w * self.config.mar_thresh_finally_opened)

        for tx, tlabel in [(t1_x, "T1 (Open)"), (t2_x, "T2 (Mid)"), (t3_x, "T3 (Peak)")]:
            cv2.line(frame, (tx, gauge_y - 3), (tx, gauge_y + 15), (255, 255, 255), 1)
            cv2.putText(frame, tlabel, (tx - 12, gauge_y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Divider line
        cv2.line(frame, (25, panel_y1 + 252), (panel_w - 15, panel_y1 + 252), (45, 45, 55), 1)

        # Stat: Cycle Timer (Active only during yawn cycle)
        elapsed = detector.get_elapsed_time()
        cv2.putText(frame, f"CYCLE TIMER ({self.config.min_yawn_duration:.1f}s - {self.config.max_yawn_duration:.1f}s)", (25, panel_y1 + 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_MUTED, 1, cv2.LINE_AA)

        if detector.current_stage != YawnStage.IDLE:
            # Color based on whether within target window
            if detector.is_completed_hold:
                timer_color = self.COLOR_GREEN
                timer_state = f"YAWN #{detector.yawn_count} CONFIRMED!"
            elif elapsed < self.config.min_yawn_duration:
                timer_color = self.COLOR_CYAN
                timer_state = f"OPENING PHASE ({elapsed:.1f}s)"
            elif elapsed <= self.config.max_yawn_duration:
                timer_color = self.COLOR_YELLOW
                timer_state = f"YAWN IN PROGRESS ({elapsed:.1f}s)"
            else:
                timer_color = self.COLOR_RED
                timer_state = f"TIMEOUT (> {self.config.max_yawn_duration:.1f}s)"

            cv2.putText(frame, f"{elapsed:.2f} s", (25, panel_y1 + 312),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, timer_color, 2, cv2.LINE_AA)
            cv2.putText(frame, timer_state, (25, panel_y1 + 332),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, timer_color, 1, cv2.LINE_AA)

            # Timer progress bar
            timer_bar_y = panel_y1 + 342
            cv2.rectangle(frame, (gauge_x1, timer_bar_y), (gauge_x2, timer_bar_y + 8), (35, 35, 45), -1)
            # Safe zone highlight on progress bar
            sz_x1 = int(gauge_x1 + gauge_w * (self.config.min_yawn_duration / self.config.max_yawn_duration))
            sz_x2 = gauge_x2
            cv2.rectangle(frame, (sz_x1, timer_bar_y), (sz_x2, timer_bar_y + 8), (40, 75, 45), -1)

            t_fill_w = int(gauge_w * min(1.0, elapsed / self.config.max_yawn_duration))
            cv2.rectangle(frame, (gauge_x1, timer_bar_y), (gauge_x1 + t_fill_w, timer_bar_y + 8), timer_color, -1)
        else:
            cv2.putText(frame, "0.00 s", (25, panel_y1 + 312),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, self.COLOR_MUTED, 2, cv2.LINE_AA)
            cv2.putText(frame, "Awaiting mouth opening...", (25, panel_y1 + 332),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Divider line
        cv2.line(frame, (25, panel_y1 + 360), (panel_w - 15, panel_y1 + 360), (45, 45, 55), 1)

        # Baseline info
        cv2.putText(frame, f"Baseline MAR: {detector.baseline_mar:.2f}", (25, panel_y1 + 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # 3D Head Pose (Yaw, Pitch, Roll)
        if landmark_data and "yaw" in landmark_data:
            yaw_deg = landmark_data.get("yaw", 0.0)
            pitch_deg = landmark_data.get("pitch", 0.0)
            roll_deg = landmark_data.get("roll", 0.0)
            pose_text = f"Pose: Y {yaw_deg:+.0f}\xb0 | P {pitch_deg:+.0f}\xb0 | R {roll_deg:+.0f}\xb0"
            pose_color = self.COLOR_CYAN if (abs(yaw_deg) < 30 and abs(pitch_deg) < 25) else self.COLOR_YELLOW
            cv2.putText(frame, pose_text, (25, panel_y1 + 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, pose_color, 1, cv2.LINE_AA)

        if detector.last_yawn_duration > 0:
            cv2.putText(frame, f"Last Yawn: {detector.last_yawn_duration:.2f}s", (25, panel_y1 + 422),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GREEN, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. BIG CONFIRMATION BANNER ON YAWN DETECTED
        # -------------------------------------------------------------
        if detector.alert_active:
            banner_h = 75
            banner_y1 = h // 2 - banner_h // 2
            banner_y2 = banner_y1 + banner_h
            banner_x1 = panel_w + 30
            banner_x2 = w - 40

            # Glowing alert box
            self._draw_translucent_box(frame, (banner_x1, banner_y1), (banner_x2, banner_y2),
                                      (20, 90, 30), alpha=0.90, radius=12)
            self._draw_rounded_rect(frame, (banner_x1, banner_y1), (banner_x2, banner_y2),
                                    self.COLOR_GREEN, radius=12, thickness=3)

            alert_title = f"*** YAWN #{detector.yawn_count} DETECTED! ***"
            (tw, th), _ = cv2.getTextSize(alert_title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            cv2.putText(frame, alert_title, (banner_x1 + (banner_x2 - banner_x1 - tw) // 2, banner_y1 + 38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            sub_text = f"Confirmed in {detector.last_yawn_duration:.2f}s | Peak MAR: {detector.peak_mar:.2f}"
            (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, sub_text, (banner_x1 + (banner_x2 - banner_x1 - stw) // 2, banner_y1 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_GREEN, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 5. BOTTOM STATUS & CONTROLS HELPER BAR
        # -------------------------------------------------------------
        bot_y1 = h - 50
        bot_y2 = h - 10
        self._draw_translucent_box(frame, (10, bot_y1), (w - 10, bot_y2), (15, 15, 20), alpha=0.85, radius=8)
        self._draw_rounded_rect(frame, (10, bot_y1), (w - 10, bot_y2), (50, 50, 65), radius=8, thickness=1)

        # Status text
        status_msg = f"STATUS: {detector.last_status_message}"
        cv2.putText(frame, status_msg, (25, bot_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Keyboard shortcuts
        shortcuts = "[C] Calib  |  [V] Rec  |  [M] Audio  |  [R] Reset  |  [S] Snap  |  [Q] Exit"
        (sw, _), _ = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, shortcuts, (w - sw - 25, bot_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_CYAN, 1, cv2.LINE_AA)
