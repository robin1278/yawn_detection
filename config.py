"""
Configuration parameters for MediaPipe FaceMesh Yawn Detection.
"""

from dataclasses import dataclass

@dataclass
class DetectorConfig:
    # --- Mouth Aspect Ratio (MAR) Thresholds ---
    # Resting / Closed threshold: mouth returned to relaxed closed state
    mar_thresh_closed: float = 0.18

    # Stage 1: Mouth starts opening beyond normal resting/speaking level
    mar_thresh_opening: float = 0.28
    
    # Mid-range reference threshold for closing transition / speech filtering
    mar_thresh_mid: float = 0.45
    
    # Stage 2: Peak yawn aperture (mouth fully opening)
    mar_thresh_finally_opened: float = 0.65
    
    # --- Duration Constraints (in seconds) ---
    # 4-stage lifecycle:
    # 1. Mouth opening increases
    # 2. Mouth fully opening (peak held)
    # 3. Mouth closing
    # 4. Mouth closing more (returns to closed baseline)
    # Accommodates human physiology: 2.5s (fast/suppressed) to 10.5s (deep/extended)
    min_yawn_duration: float = 2.4
    max_yawn_duration: float = 10.5
    # Discrete video frame sampling tolerance (~9 frames at 30fps)
    duration_tolerance: float = 0.30
    
    # --- Smoothing & Stability ---
    # Exponential moving average smoothing factor for MAR (0.0 to 1.0)
    # Higher = more responsive, Lower = smoother
    mar_smoothing_factor: float = 0.70
    
    # Minimum time in seconds mouth must hold peak aperture (distinguishes yawns from speech)
    min_peak_hold_duration: float = 0.35
    
    # Maximum time in seconds to hold Stage 2 (Fully Opening) before timeout
    max_peak_hold_duration: float = 7.5
    
    # Maximum time in seconds between stage transitions before idle timeout
    stage_timeout_duration: float = 4.0
    
    # --- Camera & Display ---
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps_target: int = 30
    
    # --- Audio Alerts ---
    enable_audio_alert: bool = True
    audio_frequency_hz: int = 880
    audio_duration_ms: int = 350
    
    # --- Landmark Indices (MediaPipe 468 Face Mesh) ---
    # Inner Lips
    LIP_INNER_TOP = 13
    LIP_INNER_BOTTOM = 14
    LIP_INNER_TOP_LEFT = 82
    LIP_INNER_BOTTOM_LEFT = 87
    LIP_INNER_TOP_RIGHT = 312
    LIP_INNER_BOTTOM_RIGHT = 317
    
    # Outer Lips
    LIP_OUTER_TOP = 0
    LIP_OUTER_BOTTOM = 17
    LIP_CORNER_LEFT = 61
    LIP_CORNER_RIGHT = 291
    
    # Facial scale references
    FOREHEAD = 10
    CHIN = 152
    NOSE_TIP = 1
