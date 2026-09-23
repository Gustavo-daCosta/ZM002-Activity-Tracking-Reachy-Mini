# Running on the physical robot

Everything under `robot/` needs the real Reachy Mini. Host, preflight and the safety rules are in
[../AGENTS.md](../AGENTS.md); the one-command demo is in [demo.md](demo.md).

## From simulation to the real robot

The same code runs on the physical robot, where the pose model gets the **robot camera** instead of the webcam and
the **physical antennas** react:

```bash
robot/deploy.sh                                       # sets up ~/wave_env on the robot and syncs the code
ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.wave_antennas --seconds 60'
```

`robot/apps/wave_antennas.py` is headless by default and imports the same `core.motion` /
`core.pose_backends` code; `core/frames.py` provides the robot frame source. Two things differ on the
robot: the pose model is **MoveNet Lightning** (`--model movenet-lightning`, ONNX) because the mediapipe aarch64
binaries abort on the CM4's Cortex-A72 (no AES instructions), and the classifier runs from
`models/wave_classifier.npz` through `motion/forest.py` (plain numpy, no scikit-learn, ~30 ms per evaluation
cheaper than `predict_proba`). `--follow` makes the robot track the person's body: the image error feeds the *rate* of the gaze (an
integrating controller, which actually centers the person, unlike a proportional one) and the body rotates to
extend the head's reach. Measured on the robot: the head pose is **absolute** (base frame) — rotating the body
makes the IK counter-rotate the head and the camera stays put, so the body does not add to the gaze, it only
gives the head (limited to ±60° from the body) somewhere further to look. Tuning: `--follow-gain`,
`--deadzone`, `--max-yaw`, `--max-head-offset`, `--body-gain`, `--max-body-yaw` (`--body-gain 0` keeps the body
still). Reach measured: ±75° of gaze against ±35° with the head alone.

`--stream-port 8080` serves the annotated camera view at `http://<robot>:8080/` (MJPEG), so the robot can run
headless while you watch what it sees from any browser. `--wave-seconds` / `--cooldown` set how long the
antennas wave and how soon another wave is answered (1.2 s + 0.5 s by default, so waving repeatedly is answered
about every 1.7 s).

The antennas are driven by their own 50 Hz thread (`motion/sender.py`): the vision loop is far too slow to
sample the wave smoothly and the movement looked stepped. Measured on the robot: **8.7 FPS**, pose 34 ms
(MoveNet int8 on TFLite/XNNPACK), motion 3 ms. The remaining limit is the camera, which hands the SDK about
10 frames per second. Step-by-step deployment notes, wheel availability for
### Package availability on aarch64 / cp312, verified on PyPI 2026-09-21

| Package | aarch64 / cp312 | Note |
|---|---|---|
| `mediapipe` | 1.0.1 installs, **but cannot run** | Measured 2026-09-21 on the robot: `FATAL ERROR: This binary was compiled with aes enabled, but this feature is not available on this processor`. The CM4's Cortex-A72 exposes only `fp asimd evtstrm crc32 cpuid` — no AES/SHA. 0.10.35 has no aarch64 wheel at all. **BlazePose is therefore impossible on this robot.** |
| `onnxruntime==1.27.0` | ✓ (and already used by the daemon) | runs the replacement pose model |
| `scikit-learn==1.9.1` | ✓ | must match the version that wrote the joblib pickle |
| `joblib`, `numpy`, `scipy`, `opencv-contrib-python` | ✓ | pulled in as dependencies |
| `rustypot` (reachy_mini dep) | ✓ (manylinux_2_17) | fine when pip runs on the robot |
| `gstreamer_cli` (via `gstreamer_bundle`) | ✗ no Linux wheel | so `reachy_mini` **cannot be pre-downloaded from the Mac**; on Linux its markers differ, so a native `pip install` on the robot should work — otherwise fall back to the apps venv `.pth` (handled by the script, step 5). |

The practical consequence of the first row is the whole reason `core/pose_backends/` exists: the robot and
this Mac cannot run the same pose model, so the backend is selected per environment behind one interface.

