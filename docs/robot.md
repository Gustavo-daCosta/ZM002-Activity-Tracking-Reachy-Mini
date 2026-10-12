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
`core.pose_backends` code; `RobotCameraSource` in `core/vision.py` provides the robot frames. Two things differ on
the robot: the pose model is **MoveNet Lightning int8** (`--model movenet-tflite`) because the mediapipe aarch64
binaries abort on the CM4's Cortex-A72 (no AES instructions), and the classifier runs from
`core/models/wave_classifier_ntu.npz` through `core/motion/forest.py` (plain numpy, no scikit-learn, ~30 ms per
evaluation cheaper than `predict_proba`). `--follow` makes the robot track the person's body: the image error feeds the *rate* of the gaze (an
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

The antennas are driven by their own 50 Hz thread (`TargetSender` in `core/motion/antennas.py`): the vision loop is far too slow to
sample the wave smoothly and the movement looked stepped. Measured on the robot: **8.7 FPS**, pose 34 ms
(MoveNet int8 on TFLite/XNNPACK), motion 3 ms. The remaining limit is the camera, which hands the SDK about
10 frames per second. Step-by-step deployment notes, wheel availability for
### Package availability on aarch64 / cp312, verified on PyPI 2026-09-21

| Package | aarch64 / cp312 | Note |
|---|---|---|
| `mediapipe` | 1.0.1 installs, **but cannot run** | Measured 2026-09-21 on the robot: `FATAL ERROR: This binary was compiled with aes enabled, but this feature is not available on this processor`. The CM4's Cortex-A72 exposes only `fp asimd evtstrm crc32 cpuid` — no AES/SHA. 0.10.35 has no aarch64 wheel at all. **BlazePose is therefore impossible on this robot.** |
| `onnxruntime==1.27.0` | ✓ (and already used by the daemon) | runs the replacement pose model |
| `scikit-learn==1.9.1` | ✓ | not installed: the robot runs the `.npz` forest export |
| `numpy`, `scipy`, `opencv-contrib-python` | ✓ | pulled in as dependencies |
| `rustypot` (reachy_mini dep) | ✓ (manylinux_2_17) | fine when pip runs on the robot |
| `gstreamer_cli` (via `gstreamer_bundle`) | ✗ no Linux wheel | so `reachy_mini` **cannot be pre-downloaded from the Mac**; on Linux its markers differ, so a native `pip install` on the robot should work — otherwise fall back to the apps venv `.pth` (handled by the script, step 5). |

The practical consequence of the first row is the whole reason `core/pose_backends/` exists: the robot and
this Mac cannot run the same pose model, so the backend is selected per environment behind one interface.

## Coach: the robot learns a child's exercises

```bash
robot/deploy.sh   # ships training/, the exercise model and the group's recordings (~/coach_data/)
ssh -t reachy 'cd ~/wave_app && ~/camera_env/bin/python -m robot.apps.coach --name Ana --stream-port 8080 \
    --train-python ~/wave_env/bin/python'
```

The robot introduces itself, guides the child through about 4.5 minutes of recording (squats, arm raises,
side bends), trains on the robot's own CPU (scikit-learn in a niced subprocess; the head keeps following),
then plays a workout with the new model. It runs in `~/camera_env` (Picamera2 needs the system Python
3.13); only the training subprocess uses `~/wave_env`, which has scikit-learn. Watch the camera with the HUD at `http://<robot>:8080/`.
Recordings (keypoints only) and models stay in `~/coach_data/` on the robot; delete a child's data with
`rm -r ~/coach_data/recordings/<name> ~/coach_data/models/<name>.*`. `--workout-only` replays the workout
with a saved model.

## Remote GPU pose inference with personalized activity recognition

In this setup, the laptop runs MoveNet pose inference on its NVIDIA GPU, then sends COCO-17 keypoints to the
Reachy. The Reachy runs the activity classifier, repetition counter, `SessionEngine`, spoken feedback and
head/body tracking. `robot.apps.remote_session_inference` does **not** load the activity classifier; the model
is selected by `robot.remote_camera_server` on the Reachy.

### Use the model trained by the Coach app

Running `robot.apps.coach --name Ana` records guided examples for `none`, `squat`, `arm_raise` and `side_bend`,
then trains a person-aware random forest. Its default output path is:

```bash
/home/pollen/coach_data/models/ana.npz
```

The name is normalized to lowercase. Check the file on the Reachy:

```bash
ls -lh ~/coach_data/models/ana.npz
```

If it does not exist, run the Coach recording/training flow first (it also starts its local workout when
training completes):

```bash
cd ~/wave_app
~/camera_env/bin/python -m robot.apps.coach \
  --name Ana \
  --stream-port 8080 \
  --train-python ~/wave_env/bin/python
```

The resulting personalized model recognizes only its trained classes: `none`, `squat`, `arm_raise` and
`side_bend`. Use `squat,arm_raise,side_bend` as the session rounds. It cannot classify `clapping`,
`jumping_jacks` or `wave`; use the default shared six-class model for those, or collect training data and fit a
model that includes those labels.

Start the server on the Reachy and explicitly select the Coach model:

```bash
cd ~/reachy-source
source ~/camera_env/bin/activate
python -m robot.remote_camera_server \
  --host localhost \
  --width 192 \
  --height 108 \
  --session \
  --follow \
  --activity-model ~/coach_data/models/ana.npz \
  --rounds squat,arm_raise,side_bend \
  --reps-per-round 4 \
  --timeout 60
```

Then run the pose-inference client on the Ubuntu laptop as usual:

```bash
cd ~/ZM002-Activity-Tracking-Reachy-Mini
source reachy_mini_env/bin/activate
python -m robot.apps.remote_session_inference \
  --model movenet-lightning \
  --remote-camera http://10.42.0.66:8765/stream.mjpg \
  --result-url http://10.42.0.66:8766/result \
  --width 192 \
  --window
```

To confirm the model selection and inspect its current predictions, query the Reachy's status endpoint:

```bash
curl -s http://10.42.0.66:8766/status
```

It reports `activity_model`, `activity_classes`, `detected_action`, `detected_confidence`, and
`action_probabilities`. During an active round, the `action` field in the session state is the *target* action;
`detected_action` is the model's current prediction.

Head/body tracking is enabled by default during active rounds. Use `--no-follow` to disable it. Tracking pauses
during welcome, round-intro, feedback and outro gestures so the follower does not overwrite gesture targets.
