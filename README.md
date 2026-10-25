# Real-Time Yawn Detection with MediaPipe FaceMesh

A real-time yawn detection system powered by **MediaPipe FaceMesh** and OpenCV, tracking the 6-stage physiological progression arc within a **4.0s - 7.0s** duration window.

---

## 🎯 Detection Logic & Progression Arc

A true yawn follows a multi-phase physiological sequence:
1. **Mouth Opening** (`Stage 1`): Lips begin separating past resting baseline.
2. **Mouth Opening More** (`Stage 2`): Aperture widens significantly beyond normal conversational speech.
3. **Mouth Finally Opened** (`Stage 3`): Peak yawn aperture (acme), held open under involuntary muscle tension.
4. **Mouth Closing** (`Stage 4`): Lips begin active contraction downwards from peak.
5. **Mouth Closing More** (`Stage 5`): Contraction continues toward resting position.
6. **Mouth Finally Closed** (`Stage 6`): Lips return to resting closed state.

**Timing Constraint**: The total cycle duration from initial opening to full closure must fall within **4.0 to 7.0 seconds**.
- Faster movements (< 4.0s) such as quick speech, laughter, smiling, or gasps are rejected.
- Prolonged open mouth (> 7.0s) such as holding mouth open or talking continuously times out and resets.
- Shallow openings (< peak MAR threshold) are rejected.

---

## 🚀 How to Run

Activate your virtual environment and run:

### 1. Live Webcam Mode
```bash
.\venv\Scripts\python.exe main.py
```
*(Optionally select another camera index with `--camera 1`)*

### 2. Live Webcam Mode with Continuous Recording
To automatically record your entire processed session (with all HUD overlays, lip meshes, and yawn alert banners) into an MP4 file in `recordings/`:
```bash
.\venv\Scripts\python.exe main.py --record
```

### 3. Interactive Demo Simulation Mode
If you want to view the full detection HUD and trigger simulated yawns with `[SPACE]`:
```bash
.\venv\Scripts\python.exe main.py --demo
```

### 4. Run on a Pre-recorded Video File
```bash
.\venv\Scripts\python.exe main.py --video path/to/video.mp4
```

### 5. Run Automated Test Suite
```bash
.\venv\Scripts\python.exe test_yawn_simulation.py
```

---

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| `[C]` | **Calibrate Baseline**: Calibrates your individual resting mouth aspect ratio (2 seconds). |
| `[V]` | **Toggle Video Recording**: Start / Stop recording the live processed video to `recordings/`. |
| `[M]` | **Toggle Sound**: Enable / Mute non-blocking audio beep on confirmed yawn. |
| `[R]` | **Reset**: Clears yawn counter and historical event records. |
| `[S]` | **Snapshot**: Saves a full-resolution timestamped screenshot into `snapshots/`. |
| `[SPACE]` | **Trigger Yawn** *(Demo mode only)*: Immediately initiates a synthetic yawn cycle. |
| `[Q]` or `[ESC]` | **Quit Application**. |

---

## 📁 Project Structure

- `config.py` — Central parameters for MAR thresholds, duration constraints (4.0s-7.0s), camera resolution, and facial landmark indices.
- `face_mesh_detector.py` — MediaPipe FaceMesh wrapper extracting 468 landmark coordinates and calculating vertical/horizontal Mouth Aspect Ratio (MAR).
- `yawn_tracker.py` — Finite State Machine tracking the 6 progressive stages, dynamic calibration, duration timing, and audio alert dispatch.
- `ui_overlay.py` — HUD overlay rendering real-time phase indicators, MAR gauge bar, live cycle timer, and confirmation banners.
- `main.py` — Application entry point supporting live camera, video playback, and interactive demo simulation.
- `test_yawn_simulation.py` — Automated unit test suite validating all 6 scenarios against ground truth constraints.
