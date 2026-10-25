"""
Video File Batch Processor for MediaPipe FaceMesh Yawn Detection.

Processes an input video file, executes 6-stage progression tracking,
renders HUD overlay on each frame, exports an annotated output video,
and generates a structured analysis report with key snapshot images.
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import Dict, Any, List

from config import DetectorConfig
from face_mesh_detector import FaceMeshDetector
from yawn_tracker import YawnDetector, YawnStage
from ui_overlay import UIOverlay


def process_video_file(video_path: str, output_path: str, snapshots_dir: str):
    if not os.path.exists(video_path):
        print(f"[ERROR] Input video not found: {video_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps

    print("=" * 68)
    print("  PROCESSING VIDEO FOR YAWN DETECTION")
    print("=" * 68)
    print(f"Input:       {video_path}")
    print(f"Resolution:  {width}x{height}")
    print(f"Framerate:   {fps:.2f} fps")
    print(f"Frames:      {total_frames}")
    print(f"Duration:    {total_duration:.2f} s")
    print(f"Output:      {output_path}")
    print("-" * 68)

    config = DetectorConfig(
        frame_width=width,
        frame_height=height,
        enable_audio_alert=False # No audio in batch video export
    )

    mesh_detector = FaceMeshDetector(config)
    yawn_detector = YawnDetector(config)
    ui_overlay = UIOverlay(config)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Event tracking
    stage_history = []
    yawn_events = []
    frame_metrics = [] # (frame_idx, timestamp, mar, stage)
    saved_snapshots = {}

    prev_stage = YawnStage.IDLE
    frame_idx = 0
    start_proc_time = time.time()

    # Pre-calibrate baseline on initial resting frames (first ~45 frames / 1.5s)
    initial_mars = []
    initial_chin_ratios = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        raw_mar, landmark_data = mesh_detector.process(frame)
        has_face = (raw_mar is not None)

        if has_face:
            if frame_idx < 45:
                initial_mars.append(raw_mar)
                if landmark_data and "jaw_ratio" in landmark_data:
                    initial_chin_ratios.append(landmark_data["jaw_ratio"])
                if frame_idx == 44 and initial_mars:
                    baseline = float(np.median(initial_mars))
                    yawn_detector.baseline_mar = max(0.12, min(baseline, 0.22))
                    if initial_chin_ratios:
                        mesh_detector.baseline_lip_chin_ratio = float(np.median(initial_chin_ratios))
                        yawn_detector.baseline_lip_chin_ratio = mesh_detector.baseline_lip_chin_ratio
                    config.mar_thresh_closed = max(0.16, yawn_detector.baseline_mar + 0.06)
                    config.mar_thresh_opening = max(0.30, yawn_detector.baseline_mar + 0.16)
                    config.mar_thresh_mid = max(0.44, yawn_detector.baseline_mar + 0.30)
                    config.mar_thresh_finally_opened = max(0.65, yawn_detector.baseline_mar + 0.48)
                    print(f"[INFO] Auto-calibrated initial baseline MAR: {yawn_detector.baseline_mar:.3f} | Chin Ratio: {mesh_detector.baseline_lip_chin_ratio:.3f}")
                    print(f"       Thresholds -> Closed: {config.mar_thresh_closed:.2f} | Open: {config.mar_thresh_opening:.2f} | Mid: {config.mar_thresh_mid:.2f} | Peak: {config.mar_thresh_finally_opened:.2f}")

            stage, is_new_yawn = yawn_detector.update(raw_mar, timestamp, landmark_data=landmark_data)

            # Log stage transitions
            if stage != prev_stage:
                print(f"  [t={timestamp:5.2f}s | Frame {frame_idx:4d}] Transition: {prev_stage.name} -> {stage.name} (MAR: {raw_mar:.3f})")
                stage_history.append((timestamp, frame_idx, prev_stage, stage, raw_mar))
                prev_stage = stage

                # Capture key stage snapshot
                if stage.name not in saved_snapshots or stage == YawnStage.MOUTH_FINALLY_OPENED:
                    snap_path = os.path.join(snapshots_dir, f"stage_{stage.value}_{stage.name.lower()}.jpg")
                    snap_frame = frame.copy()
                    stage_color = ui_overlay.get_stage_color(stage)
                    mesh_detector.draw_mouth_mesh(snap_frame, landmark_data, stage_color)
                    ui_overlay.draw_hud(snap_frame, yawn_detector, fps, has_face, landmark_data)
                    cv2.imwrite(snap_path, snap_frame)
                    saved_snapshots[stage.name] = snap_path

            if is_new_yawn:
                print(f"  *** YAWN CONFIRMED at t={timestamp:.2f}s! Duration: {yawn_detector.last_yawn_duration:.2f}s | Peak MAR: {yawn_detector.peak_mar:.3f} ***")
                yawn_events.append({
                    "timestamp": timestamp,
                    "frame": frame_idx,
                    "duration": yawn_detector.last_yawn_duration,
                    "peak_mar": yawn_detector.peak_mar,
                    "stage_durations": dict(yawn_detector.stage_durations)
                })
                # Save confirmed yawn banner frame
                banner_path = os.path.join(snapshots_dir, f"yawn_confirmed_{int(timestamp)}s.jpg")
                snap_frame = frame.copy()
                stage_color = ui_overlay.get_stage_color(stage)
                mesh_detector.draw_mouth_mesh(snap_frame, landmark_data, stage_color)
                ui_overlay.draw_hud(snap_frame, yawn_detector, fps, has_face, landmark_data)
                cv2.imwrite(banner_path, snap_frame)
                saved_snapshots["CONFIRMED_YAWN"] = banner_path

            # Draw lips overlay and HUD on output frame
            stage_color = ui_overlay.get_stage_color(stage)
            mesh_detector.draw_mouth_mesh(frame, landmark_data, stage_color)
            frame_metrics.append((frame_idx, timestamp, raw_mar, stage.name))
        else:
            if yawn_detector.current_stage != YawnStage.IDLE:
                elapsed = yawn_detector.get_elapsed_time(timestamp)
                if elapsed > config.max_yawn_duration:
                    yawn_detector.update(0.0, timestamp)
            frame_metrics.append((frame_idx, timestamp, None, "NO_FACE"))

        ui_overlay.draw_hud(frame, yawn_detector, fps, has_face, landmark_data)
        out.write(frame)

        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100
            print(f"  Progress: {pct:5.1f}% ({frame_idx}/{total_frames} frames)...", end="\r")

    cap.release()
    out.release()
    mesh_detector.close()

    elapsed_proc = time.time() - start_proc_time
    proc_fps = total_frames / elapsed_proc if elapsed_proc > 0 else 0

    print(f"\nProcessing complete in {elapsed_proc:.2f}s ({proc_fps:.1f} fps)")
    print("=" * 68)
    print("  DETECTION SUMMARY")
    print("=" * 68)
    print(f"Total Yawns Detected: {len(yawn_events)}")
    for i, ev in enumerate(yawn_events, 1):
        print(f"  Yawn #{i}: Confirmed at t={ev['timestamp']:.2f}s | Duration: {ev['duration']:.2f}s | Peak MAR: {ev['peak_mar']:.3f}")
        for sname, dur in ev['stage_durations'].items():
            print(f"    - {sname}: {dur:.2f}s")

    print("-" * 68)
    print(f"Annotated Video Saved: {output_path}")
    print(f"Key Snapshots Saved in: {snapshots_dir}")
    print("=" * 68)

    return yawn_events, saved_snapshots, stage_history


if __name__ == "__main__":
    default_input = r"C:\Users\Admin\Pictures\Camera Roll\WIN_20261024_10_16_11_Pro.mp4"
    if len(sys.argv) > 1:
        default_input = sys.argv[1]

    base_name = os.path.splitext(os.path.basename(default_input))[0]
    out_video = os.path.join(os.path.dirname(__file__), "output", f"annotated_{base_name}.mp4")
    snaps = os.path.join(os.path.dirname(__file__), "snapshots", base_name)

    process_video_file(default_input, out_video, snaps)
