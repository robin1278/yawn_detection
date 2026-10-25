"""
Main Application for Real-Time Yawn Detection using MediaPipe Face Mesh.

Detects yawns based on the 4 progressive phases:
1. Mouth Opening Increases
2. Mouth Fully Opening (peak held)
3. Mouth Closing
4. Mouth Closing More (returns to closed baseline)
All within an anatomically validated 2.5 to 10.5 seconds duration window.
"""

import sys
import time
import argparse
import os
import cv2
import numpy as np

from config import DetectorConfig
from face_mesh_detector import FaceMeshDetector
from yawn_tracker import YawnDetector
from ui_overlay import UIOverlay


def find_camera(preferred_index: int = 0) -> cv2.VideoCapture:
    """Attempt to open preferred camera, or search available indices."""
    cap = cv2.VideoCapture(preferred_index)
    if cap.isOpened():
        return cap

    # Search other indices
    for idx in [1, 2, 0]:
        if idx == preferred_index:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Camera opened on index {idx}")
            return cap
    return cap


def run_demo_mode(config: DetectorConfig, yawn_detector: YawnDetector, ui_overlay: UIOverlay):
    """Run an interactive synthetic demo simulation with live graphics and HUD."""
    window_name = "Yawn Detection - MediaPipe FaceMesh [DEMO MODE]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.frame_width, config.frame_height)

    w = config.frame_width
    h = config.frame_height
    fps = 30.0
    clock = time.time()

    # 5.2s natural yawn curve within 8.5s cycle
    cycle_period = 8.5
    yawn_duration = 5.2
    peak_mar = 0.78
    baseline_mar = 0.11

    print("[INFO] Running in Interactive Demo Mode.")
    print("       Press [SPACE] to trigger a yawn on demand.")
    print("       Press [C] to Calibrate | [M] Audio | [R] Reset | [S] Snap | [Q] Exit")

    auto_cycle = True
    manual_yawn_start = -1.0
    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    try:
        while True:
            start_loop = time.time()
            elapsed_total = start_loop - clock

            # Compute simulated MAR
            if manual_yawn_start > 0:
                yawn_t = start_loop - manual_yawn_start
                if yawn_t <= yawn_duration:
                    sine_w = np.sin(np.pi * (yawn_t / yawn_duration))
                    current_mar = baseline_mar + (peak_mar - baseline_mar) * sine_w
                else:
                    current_mar = baseline_mar
                    manual_yawn_start = -1.0
            elif auto_cycle:
                t_in_cycle = elapsed_total % cycle_period
                if t_in_cycle < yawn_duration:
                    sine_w = np.sin(np.pi * (t_in_cycle / yawn_duration))
                    current_mar = baseline_mar + (peak_mar - baseline_mar) * sine_w
                else:
                    current_mar = baseline_mar
            else:
                current_mar = baseline_mar

            # Update state machine
            stage, is_new_yawn = yawn_detector.update(current_mar, start_loop)
            stage_color = ui_overlay.get_stage_color(stage)

            # Synthetic dark studio canvas
            frame = np.full((h, w, 3), (24, 25, 32), dtype=np.uint8)

            # Animated stylized facial landmarks
            face_cx = int(w * 0.58)
            face_cy = int(h * 0.52)
            face_w = 210
            face_h = 280

            # Face oval
            cv2.ellipse(frame, (face_cx, face_cy), (face_w, face_h), 0, 0, 360, (38, 40, 50), -1)
            cv2.ellipse(frame, (face_cx, face_cy), (face_w, face_h), 0, 0, 360, (65, 70, 85), 2, lineType=cv2.LINE_AA)

            # Eyes
            eye_y = face_cy - 70
            cv2.ellipse(frame, (face_cx - 75, eye_y), (35, 12), 0, 0, 360, (55, 60, 75), -1)
            cv2.ellipse(frame, (face_cx + 75, eye_y), (35, 12), 0, 0, 360, (55, 60, 75), -1)
            cv2.circle(frame, (face_cx - 75, eye_y), 7, (200, 210, 225), -1)
            cv2.circle(frame, (face_cx + 75, eye_y), 7, (200, 210, 225), -1)

            # Nose
            nose_y = face_cy + 10
            cv2.line(frame, (face_cx, face_cy - 20), (face_cx - 10, nose_y + 15), (80, 85, 100), 2)
            cv2.line(frame, (face_cx - 10, nose_y + 15), (face_cx + 10, nose_y + 15), (80, 85, 100), 2)

            # Mouth coordinates
            mouth_y = face_cy + 110
            mouth_hw = 90
            vert_h = int(current_mar * 2 * mouth_hw)
            vert_half = max(4, vert_h // 2)

            p_top = (face_cx, mouth_y - vert_half)
            p_bot = (face_cx, mouth_y + vert_half)
            p_left = (face_cx - mouth_hw, mouth_y)
            p_right = (face_cx + mouth_hw, mouth_y)

            # Outer and inner lip polygons
            angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            inner_pts = []
            outer_pts = []
            for ang in angles:
                ix = int(face_cx + mouth_hw * np.cos(ang))
                iy = int(mouth_y + vert_half * np.sin(ang))
                inner_pts.append((ix, iy))
                ox = int(face_cx + (mouth_hw + 14) * np.cos(ang))
                oy = int(mouth_y + (vert_half + 12) * np.sin(ang))
                outer_pts.append((ox, oy))

            cv2.fillPoly(frame, [np.array(inner_pts, dtype=np.int32)], (15, 12, 18))
            cv2.polylines(frame, [np.array(outer_pts, dtype=np.int32)], True, (130, 130, 140), 1, cv2.LINE_AA)
            cv2.polylines(frame, [np.array(inner_pts, dtype=np.int32)], True, stage_color, 2, cv2.LINE_AA)

            cv2.line(frame, p_top, p_bot, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(frame, p_left, p_right, (255, 200, 0), 1, lineType=cv2.LINE_AA)
            for pt in [p_top, p_bot, p_left, p_right]:
                cv2.circle(frame, pt, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            # Render HUD overlay
            ui_overlay.draw_hud(frame, yawn_detector, fps, has_face=True)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Exiting demo mode...")
                break
            elif key == 32:  # SPACE
                manual_yawn_start = time.time()
                auto_cycle = False
                print("[INFO] Manual yawn cycle triggered!")
            elif key == ord('c') or key == ord('C'):
                yawn_detector.start_calibration()
                print("[INFO] Baseline calibration started.")
            elif key == ord('m') or key == ord('M'):
                config.enable_audio_alert = not config.enable_audio_alert
                print(f"[INFO] Audio alerts: {'Enabled' if config.enable_audio_alert else 'Disabled'}")
            elif key == ord('r') or key == ord('R'):
                yawn_detector.reset_stats()
                print("[INFO] Statistics reset.")
            elif key == ord('s') or key == ord('S'):
                filename = os.path.join(snapshots_dir, f"yawn_demo_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot saved to: {filename}")

    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MediaPipe FaceMesh Yawn Detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default="", help="Path to video file instead of live camera")
    parser.add_argument("--demo", action="store_true", help="Run animated visual simulation demo mode")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio notification")
    parser.add_argument("--record", action="store_true", help="Record and save real-time processed video stream to recordings/")
    args = parser.parse_args()

    config = DetectorConfig()
    config.camera_index = args.camera
    if args.no_audio:
        config.enable_audio_alert = False

    print("=" * 65)
    print("  YAWN DETECTION SYSTEM - MEDIAPIPE FACE MESH")
    print("  6 Progressive Phases | 4.0s - 7.0s Target Duration Window")
    print("=" * 65)
    print("Controls:")
    print("  [C] - Calibrate resting mouth baseline")
    print("  [V] - Toggle continuous video recording on/off")
    print("  [M] - Toggle audio alert on/off")
    print("  [R] - Reset yawn counters and statistics")
    print("  [S] - Save snapshot screenshot")
    print("  [Q] - Quit application")
    print("-" * 65)

    # Initialize components
    mesh_detector = FaceMeshDetector(config)
    yawn_detector = YawnDetector(config)
    ui_overlay = UIOverlay(config)

    # Demo mode branch
    if args.demo:
        run_demo_mode(config, yawn_detector, ui_overlay)
        return

    # Open video source
    if args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] Video file not found: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        print(f"[INFO] Playing video file: {args.video}")
    else:
        print(f"[INFO] Connecting to camera (index {config.camera_index})...")
        cap = find_camera(config.camera_index)

    if not cap.isOpened():
        print("[WARNING] Could not open webcam.")
        print("[INFO] Launching interactive demo mode instead...")
        run_demo_mode(config, yawn_detector, ui_overlay)
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    window_name = "Yawn Detection - MediaPipe FaceMesh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.frame_width, config.frame_height)

    prev_time = time.time()
    fps = 30.0

    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    # Real-time video recording setup
    is_recording = args.record
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rec_filepath = ""

    if is_recording:
        rec_filepath = os.path.join(recordings_dir, f"live_session_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        video_writer = cv2.VideoWriter(rec_filepath, fourcc, 30.0, (config.frame_width, config.frame_height))
        print(f"[INFO] Continuous video recording ENABLED: {rec_filepath}")

    # Automatically calibrate resting mouth baseline on launch
    if not args.video:
        yawn_detector.start_calibration()
        print("[INFO] Automatic initial calibration started. Please keep mouth resting and closed for ~2 seconds.")

    failed_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    failed_frame_count += 1
                    if failed_frame_count > 15:
                        print("[WARNING] Failed to grab frame from camera after multiple attempts.")
                        break
                    time.sleep(0.04)
                    continue
            failed_frame_count = 0

            # Mirror frame horizontally for intuitive webcam interaction
            if not args.video:
                frame = cv2.flip(frame, 1)

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Process frame with FaceMesh
            raw_mar, landmark_data = mesh_detector.process(frame)
            has_face = (raw_mar is not None)

            if has_face:
                # Synchronize calibrated baseline to mesh detector
                mesh_detector.baseline_lip_chin_ratio = yawn_detector.baseline_lip_chin_ratio
                # Update yawn state machine
                stage, is_new_yawn = yawn_detector.update(raw_mar, curr_time, landmark_data=landmark_data)

                # Draw lips contours and measurement vectors
                stage_color = ui_overlay.get_stage_color(stage)
                mesh_detector.draw_mouth_mesh(frame, landmark_data, stage_color)
            else:
                # If face is lost, update with 0 or reset
                if yawn_detector.current_stage != yawn_detector.current_stage.IDLE:
                    elapsed = yawn_detector.get_elapsed_time(curr_time)
                    if elapsed > config.max_yawn_duration:
                        yawn_detector.update(0.0, curr_time)

            # Draw HUD overlay (including recording indicator)
            ui_overlay.draw_hud(frame, yawn_detector, fps, has_face, landmark_data, is_recording=is_recording)

            # Write frame to video recording if active
            if is_recording and video_writer is not None:
                video_writer.write(frame)

            # Display frame
            cv2.imshow(window_name, frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[INFO] Exiting application...")
                break
            elif key == ord('v') or key == ord('V'):
                is_recording = not is_recording
                if is_recording:
                    rec_filepath = os.path.join(recordings_dir, f"live_session_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                    video_writer = cv2.VideoWriter(rec_filepath, fourcc, 30.0, (config.frame_width, config.frame_height))
                    print(f"\n[INFO] >>> RECORDING STARTED: {rec_filepath}")
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print(f"\n[INFO] >>> RECORDING STOPPED. File saved to: {rec_filepath}")
            elif key == ord('c') or key == ord('C'):
                yawn_detector.start_calibration()
                print("[INFO] Baseline calibration started. Keep mouth closed.")
            elif key == ord('m') or key == ord('M'):
                config.enable_audio_alert = not config.enable_audio_alert
                print(f"[INFO] Audio alerts: {'Enabled' if config.enable_audio_alert else 'Disabled'}")
            elif key == ord('r') or key == ord('R'):
                yawn_detector.reset_stats()
                print("[INFO] Statistics reset.")
            elif key == ord('s') or key == ord('S'):
                filename = os.path.join(snapshots_dir, f"yawn_snap_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot saved to: {filename}")

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Recording finalized and saved to: {rec_filepath}")
        cap.release()
        mesh_detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Resources released successfully.")


if __name__ == "__main__":
    main()
