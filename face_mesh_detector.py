"""
MediaPipe FaceMesh Landmark Detector & Mouth Aspect Ratio (MAR) Calculator.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import mediapipe as mp
from config import DetectorConfig


class FaceMeshDetector:
    """
    Wrapper around MediaPipe FaceMesh to extract facial landmarks
    and compute Mouth Aspect Ratio (MAR).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Lip landmark indices for drawing contours
        self.INNER_LIPS = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
        self.OUTER_LIPS = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
            375, 321, 405, 314, 17, 84, 181, 91, 146
        ]

        # Lip thickness ratio relative to mouth width at rest (~26-30% of mouth width)
        # Lip thickness ratio relative to mouth width at rest (~26-30% of mouth width)
        self.lip_thickness_ratio: float = 0.28
        self.baseline_lip_chin_ratio: float = 0.60

    def _euclidean_dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two 2D points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _dist_3d(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def process(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Process a video frame and compute angle-invariant MAR in Head Local Frame.
        Returns:
            (mar, landmark_data_dict) or (None, None) if no face detected.
        """
        h, w, _ = frame.shape
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark

        def get_pt(idx: int) -> Tuple[int, int]:
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        def get_pt3(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w], dtype=np.float64)

        # -------------------------------------------------------------
        # 1. 3D HEAD LOCAL ANATOMICAL COORDINATE FRAME [u_x, u_y, u_z]
        # -------------------------------------------------------------
        # Rigid cranial landmarks: outer eye corners (33, 263) and forehead (10)
        eye_l_3d = get_pt3(33)
        eye_r_3d = get_pt3(263)
        forehead_3d = get_pt3(self.config.FOREHEAD)
        cranial_mid = (eye_l_3d + eye_r_3d) * 0.5

        # Facial Lateral Axis u_x (right to left from image perspective)
        v_x = eye_r_3d - eye_l_3d
        eye_dist_3d = max(10.0, float(np.linalg.norm(v_x)))
        u_x = v_x / eye_dist_3d

        # Facial Vertical Axis u_y (cranial vertical orthogonalized against u_x)
        v_y_cranial = cranial_mid - forehead_3d
        v_y_ortho = v_y_cranial - np.dot(v_y_cranial, u_x) * u_x
        norm_y = float(np.linalg.norm(v_y_ortho))
        u_y = v_y_ortho / (norm_y if norm_y > 1e-6 else 1.0)

        # Facial Normal Axis u_z (pointing outward from face)
        u_z = np.cross(u_x, u_y)
        norm_z = float(np.linalg.norm(u_z))
        u_z = u_z / (norm_z if norm_z > 1e-6 else 1.0)

        # Instantaneous head pose angles (degrees)
        yaw = float(np.degrees(np.arctan2(u_x[2], u_x[0])))
        pitch = float(np.degrees(np.arctan2(u_y[2], u_y[1])))
        roll = float(np.degrees(np.arctan2(u_x[1], u_x[0])))

        # -------------------------------------------------------------
        # 2. KEY 3D MOUTH LANDMARKS
        # -------------------------------------------------------------
        # Inner lips (3D)
        p_top_inner_3d = get_pt3(self.config.LIP_INNER_TOP)          # 13
        p_bottom_inner_3d = get_pt3(self.config.LIP_INNER_BOTTOM)    # 14
        p_top_left_inner_3d = get_pt3(self.config.LIP_INNER_TOP_LEFT)       # 82
        p_bottom_left_inner_3d = get_pt3(self.config.LIP_INNER_BOTTOM_LEFT) # 87
        p_top_right_inner_3d = get_pt3(self.config.LIP_INNER_TOP_RIGHT)     # 312
        p_bottom_right_inner_3d = get_pt3(self.config.LIP_INNER_BOTTOM_RIGHT) # 317

        # Outer lips (3D)
        p_top_outer_3d = get_pt3(self.config.LIP_OUTER_TOP)          # 0
        p_bottom_outer_3d = get_pt3(self.config.LIP_OUTER_BOTTOM)    # 17
        p_top_left_outer_3d = get_pt3(37)
        p_bottom_left_outer_3d = get_pt3(84)
        p_top_right_outer_3d = get_pt3(267)
        p_bottom_right_outer_3d = get_pt3(314)

        # Mouth corners & chin
        p_corner_left_3d = get_pt3(self.config.LIP_CORNER_LEFT)      # 61
        p_corner_right_3d = get_pt3(self.config.LIP_CORNER_RIGHT)    # 291
        p_chin_3d = get_pt3(self.config.CHIN)                        # 152

        # 2D screen positions for HUD rendering
        p_top_inner = get_pt(self.config.LIP_INNER_TOP)
        p_bottom_inner = get_pt(self.config.LIP_INNER_BOTTOM)
        p_top_outer = get_pt(self.config.LIP_OUTER_TOP)
        p_bottom_outer = get_pt(self.config.LIP_OUTER_BOTTOM)
        p_top_left = get_pt(self.config.LIP_INNER_TOP_LEFT)
        p_bottom_left = get_pt(self.config.LIP_INNER_BOTTOM_LEFT)
        p_top_right = get_pt(self.config.LIP_INNER_TOP_RIGHT)
        p_bottom_right = get_pt(self.config.LIP_INNER_BOTTOM_RIGHT)
        p_corner_left = get_pt(self.config.LIP_CORNER_LEFT)
        p_corner_right = get_pt(self.config.LIP_CORNER_RIGHT)
        p_forehead = get_pt(self.config.FOREHEAD)
        p_chin = get_pt(self.config.CHIN)

        # -------------------------------------------------------------
        # 3. ANGLE-INVARIANT VERTICAL APERTURE IN HEAD LOCAL FRAME
        # -------------------------------------------------------------
        # Vertical parting of lips is projected strictly along facial vertical axis u_y
        ap_in_c = abs(float(np.dot(p_bottom_inner_3d - p_top_inner_3d, u_y)))
        ap_in_l = abs(float(np.dot(p_bottom_left_inner_3d - p_top_left_inner_3d, u_y)))
        ap_in_r = abs(float(np.dot(p_bottom_right_inner_3d - p_top_right_inner_3d, u_y)))

        # Multi-point oblique yaw compensation:
        # When yaw > 25 degrees, the far side of the lips becomes foreshortened/occluded.
        # Emphasize the unoccluded near side and central aperture.
        if abs(yaw) > 25.0:
            near_in = ap_in_l if yaw < 0 else ap_in_r
            robust_in_vertical = max(ap_in_c, 0.7 * ap_in_c + 0.3 * near_in)
        else:
            robust_in_vertical = (ap_in_c + ap_in_l + ap_in_r) / 3.0

        # Outer lips vertical aperture
        ap_out_c = abs(float(np.dot(p_bottom_outer_3d - p_top_outer_3d, u_y)))
        ap_out_l = abs(float(np.dot(p_bottom_left_outer_3d - p_top_left_outer_3d, u_y)))
        ap_out_r = abs(float(np.dot(p_bottom_right_outer_3d - p_top_right_outer_3d, u_y)))

        if abs(yaw) > 25.0:
            near_out = ap_out_l if yaw < 0 else ap_out_r
            robust_out_vertical = max(ap_out_c, 0.7 * ap_out_c + 0.3 * near_out)
        else:
            robust_out_vertical = (ap_out_c + ap_out_l + ap_out_r) / 3.0

        # -------------------------------------------------------------
        # 4. ROBUST 3D ROTATION-INVARIANT HORIZONTAL MOUTH WIDTH
        # -------------------------------------------------------------
        mouth_delta = p_corner_right_3d - p_corner_left_3d
        w_lateral = abs(float(np.dot(mouth_delta, u_x)))
        w_3d = float(np.linalg.norm(mouth_delta))
        # Anchor mouth width to skeletal inter-ocular distance (universally >= 52% of eye distance)
        robust_width = max(w_lateral, w_3d, eye_dist_3d * 0.52, 10.0)

        # Scale-invariant outer lip thickness scaled to robust mouth width
        lip_thickness_px = self.lip_thickness_ratio * robust_width
        outer_aperture = max(0.0, robust_out_vertical - lip_thickness_px)

        # -------------------------------------------------------------
        # 5. CONDITIONAL MANDIBULAR JAW DROP TRACKING
        # -------------------------------------------------------------
        # Measured along facial vertical axis u_y
        d_chin_y = abs(float(np.dot(p_chin_3d - p_top_outer_3d, u_y)))
        jaw_ratio = d_chin_y / eye_dist_3d
        resting_chin_px = self.baseline_lip_chin_ratio * eye_dist_3d
        jaw_drop = max(0.0, d_chin_y - resting_chin_px)

        # Only activate jaw tracking as a safety net when an opening is genuinely occurring,
        # preventing teeth occlusion collapse without injecting fake aperture at rest.
        if robust_in_vertical > 12.0 or outer_aperture > 15.0 or jaw_drop > (0.18 * eye_dist_3d):
            jaw_aperture = jaw_drop * 0.80
        else:
            jaw_aperture = 0.0

        # -------------------------------------------------------------
        # 6. DUAL-LAYER APERTURE SYNTHESIS & MAR
        # -------------------------------------------------------------
        robust_vertical = max(robust_in_vertical, outer_aperture, jaw_aperture)
        mar = robust_vertical / robust_width

        face_height = max(1.0, float(np.linalg.norm(forehead_3d - p_chin_3d)))

        data = {
            "p_top_inner": p_top_inner,
            "p_bottom_inner": p_bottom_inner,
            "p_top_outer": p_top_outer,
            "p_bottom_outer": p_bottom_outer,
            "p_top_left": p_top_left,
            "p_bottom_left": p_bottom_left,
            "p_top_right": p_top_right,
            "p_bottom_right": p_bottom_right,
            "p_corner_left": p_corner_left,
            "p_corner_right": p_corner_right,
            "p_forehead": p_forehead,
            "p_chin": p_chin,
            "inner_pts": [get_pt(i) for i in self.INNER_LIPS],
            "outer_pts": [get_pt(i) for i in self.OUTER_LIPS],
            "d_vertical_px": robust_vertical,
            "d_horizontal_px": robust_width,
            "face_height_px": face_height,
            "eye_dist_3d": eye_dist_3d,
            "jaw_ratio": jaw_ratio,
            "jaw_drop_px": jaw_drop,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "inner_collapsed": bool(robust_in_vertical < 0.4 * outer_aperture and outer_aperture > 15.0),
            "normalized_aperture": robust_vertical / face_height,
            "raw_mar": mar
        }

        return mar, data

    def draw_mouth_mesh(self, frame: np.ndarray, data: Dict[str, Any], stage_color: Tuple[int, int, int]):
        """
        Draw visual feedback over mouth landmarks on the given frame.
        """
        inner_pts = np.array(data["inner_pts"], dtype=np.int32)
        outer_pts = np.array(data["outer_pts"], dtype=np.int32)

        # Draw smooth polygon contours for lips
        cv2.polylines(frame, [outer_pts], isClosed=True, color=(160, 160, 170), thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [inner_pts], isClosed=True, color=stage_color, thickness=2, lineType=cv2.LINE_AA)

        # Choose best vertical measurement vector line
        # If inner lips collapsed, show vector between outer lips to indicate true aperture
        p_top = data["p_top_inner"]
        p_bot = data["p_bottom_inner"]
        if data.get("inner_collapsed", False) or (self._euclidean_dist(p_top, p_bot) < 8.0 and self._euclidean_dist(data["p_top_outer"], data["p_bottom_outer"]) > 28.0):
            p_top = data["p_top_outer"]
            p_bot = data["p_bottom_outer"]

        cv2.line(frame, p_top, p_bot, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # Draw horizontal measurement vector line
        p_left = data["p_corner_left"]
        p_right = data["p_corner_right"]
        cv2.line(frame, p_left, p_right, (255, 200, 0), 1, lineType=cv2.LINE_AA)

        # Draw key landmark points
        for pt in [p_top, p_bot, p_left, p_right]:
            cv2.circle(frame, pt, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
