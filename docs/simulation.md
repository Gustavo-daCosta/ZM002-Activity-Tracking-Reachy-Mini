# Simulation and webcam tracking

Run Reachy Mini in a MuJoCo 3D simulation on this computer, driven by the **laptop webcam** instead of
the robot camera. No robot and no preflight needed: the daemon runs on `localhost`.


## How it works

```
 laptop webcam ──► OpenCV ──► MediaPipe Pose ──► nose (x, y) ──► yaw / pitch ──► ReachyMini.set_target()
   (FaceTime)      (mirror)    (tasks API)       in 0..1          + smoothing        │ HTTP/WebSocket
                                                                                     ▼
                                                              reachy_mini daemon --sim (localhost:8000)
                                                                     └── MuJoCo 3D viewer
```

- **The simulated daemon** is the same `reachy_mini` daemon as on the real robot, started with `--sim`.
  It runs MuJoCo physics, opens the 3D window and serves the same REST API and SDK connection on
  `http://localhost:8000`. SDK code cannot tell the difference, so what works here works on the robot
  (with the right host).
- **No camera or microphone in the simulation.** The daemon is started with `--no-media` and the SDK connects
  with `media_backend="no_media"`. The webcam is read directly with OpenCV and "pretends" to be the robot camera.
- **Tracking.** MediaPipe Pose Landmarker (VIDEO mode, CPU) finds the nose in the mirrored frame. Its offset
  from the image center becomes head angles (`tracking_math.py`):
  - nose on the right edge → `+max_yaw` (the head turns to its left, toward you); left edge → `-max_yaw`
  - nose at the bottom → `+max_pitch` (the head looks down); top → `-max_pitch`
  - a small dead zone around the center avoids jitter, and exponential smoothing avoids jerky moves
  - if nobody is detected, the head slowly drifts back to center instead of freezing
- **Motion.** `set_target()` is non-blocking and meant for reactive loops. Updates are rate-limited
  (20 Hz by default). `goto_target()` is only used to center the head at start and exit.

> This is position mapping, not closed-loop tracking. In the simulation the camera does not move with the head.
> On the real robot the camera is in the head, so the same mapping would chase its own motion. There, use the
> daemon's head tracking or a relative (error-driven) controller instead (see `robot/apps/head_tracking.py`).

## Files

Simulation-only, under `sim/`:

| File | Purpose |
|---|---|
| `sim/start.sh` | Starts the daemon in simulation (`mjpython` on macOS) with the 3D viewer |
| `sim/head_tracking.py` | Webcam -> MediaPipe -> simulated head |
| `sim/body_tracking.py` | Body tracking with `--model` selection, a performance panel and `--detect-actions` |
| `sim/pose_viewer.py` | Pose on the webcam only (no daemon), to check the vision part alone |
| `sim/camera_check.py` | Shows a webcam / lists cameras **by name** |

Shared with the robot, under `core/` -- the same code runs in both places:

| File | Purpose |
|---|---|
| `core/vision.py` | Webcam opening, pose model download, drawing helpers |
| `core/tracking_math.py` | Pure math (image -> angles, smoothing), unit tested |
| `core/body_center.py` | Target rule: torso -> shoulders -> nose |
| `core/pose_backends/` | Pose models behind one interface -- see [pose-models.md](pose-models.md) |
| `core/motion/` | Wave and action recognition -- see [action-recognition.md](action-recognition.md) |
| `core/metrics.py` | Per-stage timings, FPS and exit summary |
| `core/models/` | The `.npz` forests are tracked; pose weights download on first run (git-ignored) |

## Setup (once)

Uses the repo virtualenv `reachy_mini_env` (Python 3.12, `reachy_mini==1.10.0`):

```bash
reachy_mini_env/bin/python -m pip install -r requirements/sim.txt
```

This installs `mujoco==3.3.0` (the version pinned by `reachy_mini[mujoco]`), which provides `mjpython`,
`mediapipe==0.10.35` and `cv2_enumerate_cameras` (select the webcam by name).

- `mediapipe 1.0.1` aborts on macOS arm64 (`Check failed: service_ Service is unavailable`).
- `mediapipe 0.10.21` works, but downgrades numpy to 1.x and protobuf in the shared venv.
- 0.10.30+ no longer ships the legacy `mp.solutions` API, so this code uses `mediapipe.tasks`.

On macOS, allow camera access for your terminal / IDE (System Settings → Privacy & Security → Camera).

## Run

Terminal 1, the simulation (keep it open):

```bash
sim/start.sh
# same as: reachy_mini_env/bin/mjpython -m reachy_mini.daemon.app.main --sim --no-media
```

Wait for `Daemon started successfully.` and the 3D window. The simulated robot wakes up on start.
Check with `curl -s localhost:8000/api/daemon/status` (`"simulation_enabled": true`).

Terminal 2, the tracking:

```bash
reachy_mini_env/bin/python -m sim.head_tracking
```

Press `q` or `ESC` in the webcam window to quit. The head returns to center. Stop the simulation with `Ctrl+C`
in terminal 1.

### Options

```bash
reachy_mini_env/bin/python -m sim.head_tracking --help
  --camera FaceTime    camera name substring or OpenCV index (default FaceTime = built-in Mac camera)
  --model lite|full    pose model: lite is faster, full is more accurate
  --rate 20            max head updates per second
  --max-yaw 40         degrees at the left/right image edge
  --max-pitch 25       degrees at the top/bottom image edge (hardware limit is ±40°)
  --deadzone 0.03      ignored offset from the center (fraction of the image)
  --smoothing 0.2      0..1, higher = faster but jerkier
  --host localhost     daemon host
```

Checks without the simulation:

```bash
reachy_mini_env/bin/python -m sim.camera_check --list    # index and name of each camera
reachy_mini_env/bin/python -m sim.camera_check --camera FaceTime
reachy_mini_env/bin/python -m sim.pose_viewer
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Wrong camera (iPhone) or black image | Select by name: `--camera FaceTime` (default). OpenCV indices change when the iPhone (Continuity Camera) connects or disconnects, so avoid numeric indices; `camera_check --list` shows names |
| `No camera matching ...` | The message lists the available cameras; pass part of the right name |
| `Could not open camera` | Grant camera permission to the terminal, close apps using the webcam |
| SDK cannot connect | Is `sim/start.sh` running? `curl localhost:8000/api/daemon/status` |
| `Address already in use` on port 8000 | Another daemon is running (`lsof -iTCP:8000 -sTCP:LISTEN`) |
| MuJoCo viewer error on macOS | Start with `mjpython` (what `sim/start.sh` does), not `python` |
| Segfault from `libgstpython` | GStreamer Python plugin issue, see the simulation troubleshooting in the [Reachy Mini docs](https://huggingface.co/docs/reachy_mini/SDK/quickstart) |
| `GStreamer-WARNING ... plugin loader` | Harmless |
| `IK error: ... not achievable` | Lower `--max-pitch` / `--max-yaw` (limits: pitch/roll ±40°, head yaw vs body ±60°) |
| Slow FPS | `--model lite` (default), close other apps |

## Scenes and headless mode

Extra arguments go to the daemon: `sim/start.sh --scene minimal` loads a scene with objects,
and `--headless` runs without the 3D window.

Reference: [Reachy Mini SDK quickstart](https://huggingface.co/docs/reachy_mini/SDK/quickstart).
