# Real-Time Yawn Detection with MediaPipe FaceMesh

A real-time AI yawn detection system powered by **MediaPipe FaceMesh** (468 3D landmarks) and OpenCV, tracking the 4-stage physiological progression arc within an anatomically validated **2.5s - 10.5s** duration window.

---

## 🎯 Detection Logic & Progression Arc

A true yawn follows a multi-phase physiological sequence:
1. **Stage 1: Mouth Opening Increases**: Lips begin separating past resting baseline.
2. **Stage 2: Mouth Fully Opening (Peak Aperture)**: Wide aperture (acme) with sustained muscle dwell time (min 0.35s). Robust against jaw tremors and secondary stretch expansion.
3. **Stage 3: Mouth Closing**: Active muscular contraction down from peak.
4. **Stage 4: Mouth Closing More**: Returns to resting closed baseline (or lips touch gap <= 4.5px).

**1:1 Synchronized Detection & Alarm**:
- The confirmation banner (`*** YAWN #X DETECTED! ***`) and counter increment simultaneously at the exact frame of confirmation.
- Cycle timer progress indicator tracks in-progress duration without false alarms.

**Timing Constraint**: The total cycle duration from opening to full closure must fall within **2.5 to 10.5 seconds**.
- Faster movements (< 2.4s) such as quick speech, laughter, smiling, or gasps are rejected.
- Prolonged open mouth (> 10.5s) times out and resets.
- Shallow openings (< peak MAR threshold) are rejected.

---

## 🚀 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/robin1278/yawn_detection.git
cd yawn_detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Live Webcam Mode
```bash
python main.py
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
