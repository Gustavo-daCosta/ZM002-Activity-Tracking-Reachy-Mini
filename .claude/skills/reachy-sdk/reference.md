# Reachy Mini Python SDK — reference (`reachy_mini` 1.10.0)

Source of truth: `reachy_mini_env/lib/python3.12/site-packages/reachy_mini/` (read it when in doubt).
Behaviour marked **(live)** was verified on our wireless robot over the network (WebRTC media).

## Connection

```python
from reachy_mini import ReachyMini

ReachyMini(
    robot_name="reachy_mini",        # non-default name → verified / resolved via mDNS
    host="reachy-mini.local",        # used in "network" mode (and as auto fallback)
    port=8000,
    connection_mode="auto",          # "auto" (localhost first, then host) | "localhost_only" | "network"
    spawn_daemon=False, use_sim=False,
    timeout=5.0,
    automatic_body_yaw=True,         # body yaw follows head yaw in IK
    log_level="INFO",
    media_backend="default",         # "default" | "no_media" | "local" | "webrtc"
    localhost_only=None,             # deprecated alias
)
```

- From this Mac always use `connection_mode="network"` with the robot IP — in this project use
  `robot.connect.connect_robot()` instead of constructing `ReachyMini` directly.
- `media_backend="default"` → `LOCAL` (GStreamer IPC) only when `localhost_only` and the daemon camera socket
  exists (code running **on** the robot), otherwise `WEBRTC` (remote streaming).
- `media_backend="no_media"` → **the SDK calls `release_media()` on the daemon** (camera/mic freed for direct access
  on the robot). Daemon head tracking then gets no frames. `connect_robot(media=False)` undoes this.
- Deprecated backend names (`gstreamer`, `sounddevice_opencv`, `default_no_video`, …) map to `local` with a warning.
- A version mismatch SDK≠daemon emits `RuntimeWarning` — keep both at the same version.
- Context manager `__exit__`: re-acquires media if it was released by this instance, closes media, disconnects.
- Constructor does **not** enable motors or wake the robot.
- `reachy_mini.utils.discovery.find_robots(timeout=5.0) -> list[DiscoveredRobot]` finds daemons via mDNS.

## State reading

| Call | Returns |
|---|---|
| `get_current_head_pose()` | 4×4 `np.ndarray` (meters) |
| `get_current_joint_positions()` | `(head_joints[7], antennas[2])` rad; head = `[body yaw, stewart_1..6]` |
| `get_present_antenna_joint_positions()` | `[a0, a1]` rad |
| `imu` (property) | `{"accelerometer": [m/s²]*3, "gyroscope": [rad/s]*3, "quaternion": [w,x,y,z], "temperature": °C}` or `None` (Lite). 50 Hz cache **(live)** |
| `get_tracked_face(wait=True, timeout=5.0)` | `FaceTarget(detected, x, y, roll, ts)` |
| `media_released` (property) | whether this instance released daemon media |

## Motion

Poses are 4×4 homogeneous matrices. Build them with:

```python
from reachy_mini.utils import create_head_pose
create_head_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, mm=False, degrees=True)
# mm=True → x,y,z given in millimeters; degrees=False → angles in radians. Euler order "xyz".
```

Frame: origin at neutral head, **x forward, y left, z up**. Positive pitch looks down, positive yaw turns left.

### `goto_target(head=None, antennas=None, duration=0.5, method=InterpolationTechnique.MIN_JERK, body_yaw=0.0)`

- Interpolated move executed **by the daemon**; the call **blocks** until done (timeout `duration + 1 s`).
- `method`: `"linear" | "minjerk" | "ease_in_out" | "cartoon"` (`reachy_mini.utils.interpolation.InterpolationTechnique`).
- ⚠️ `body_yaw` defaults to `0.0` → every `goto_target` **recenters the body** unless you pass `body_yaw=None`.
- `duration <= 0` raises; use `set_target` for immediate targets.
- At least one of head/antennas/body_yaw is required.

### `set_target(head=None, antennas=None, body_yaw=None)`

- Immediate target, no interpolation, non-blocking. Use it in a loop (≈50–100 Hz) for continuous control **(live)**:

```python
t0 = time.time()
while time.time() - t0 < 3:
    t = time.time() - t0
    mini.set_target(head=create_head_pose(yaw=10 * np.sin(2 * np.pi * t)),
                    antennas=[0.3 * np.sin(4 * t), -0.3 * np.sin(4 * t)])
    time.sleep(0.02)
```

- Call it **after** `enable_motors()` (enabling pins targets to the present pose).
- Lower-level single-part setters: `set_target_head_pose(pose4x4)`, `set_target_antenna_joint_positions([a0, a1])`,
  `set_target_body_yaw(rad)`.

### Other motion

| Call | Notes |
|---|---|
| `wake_up()` | goto neutral (2 s), antennas `[-0.1745, 0.1745]`, plays `wake_up.wav` (needs media), small roll wiggle |
| `goto_sleep()` | back to neutral if far, plays `go_sleep.wav`, sleep pose (2 s) + 2 s wait. Motors stay enabled |
| `look_at_world(x, y, z, duration=1.0, perform_movement=True)` | meters in robot frame; returns the pose; `duration=0` → `set_target` **(live)** |
| `look_at_image(u, v, duration=1.0, perform_movement=True)` | pixel (u right, v down) on the camera image; needs camera + calibration (media backend active) **(live)** |
| `set_automatic_body_yaw(enabled)` | when on, body rotates to help large head yaw |
| `cancel_move()` | stops a running `play_move` loop and audio |

Body yaw only: `mini.goto_target(body_yaw=np.deg2rad(20), duration=0.8)` **(live)**.
Antennas: sleep ≈ `[-3.05, 3.05]`, neutral `[-0.1745, 0.1745]`. Order is **`[right, left]` from the robot's
point of view** — index 0 is the antenna on *your left* when facing the robot (verified visually 2026-09-14;
the REST docstring saying "(left, right)" is wrong).

Motor names for `enable_motors(ids=...)` / `disable_motors(ids=...)`: `body_rotation`, `stewart_1` … `stewart_6`,
`right_antenna`, `left_antenna`.

## Recorded moves (emotions, dances)

```python
from reachy_mini.motion.recorded_move import RecordedMoves

emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")   # 85 moves (live)
dances = RecordedMoves("pollen-robotics/reachy-mini-dances-library")       # 19 moves
emotions.list_moves()
move = emotions.get("yes1")          # RecordedMove: .description, .duration, .sound_path
mini.play_move(move, play_frequency=100.0, initial_goto_duration=0.8, sound=True)  # blocking
await mini.async_play_move(move, ...)                                              # asyncio version
```

- Datasets download from Hugging Face on the first use on this Mac (cached in `~/.cache/huggingface`).
- `play_move` streams `set_target` from the client at `play_frequency`; `sound=True` plays the sidecar
  `.wav` through `mini.media` (needs a media backend, otherwise "Audio system is not initialized").
- Alternative without local download: REST `POST /api/move/play/recorded-move-dataset/<dataset>/<move>` (runs on the daemon).
- Custom moves: subclass `reachy_mini.motion.move.Move` (`duration`, `sound_path`, `evaluate(t) -> (head4x4, antennas, body_yaw)`).
  `reachy_mini.motion.goto.GotoMove` is the built-in interpolated implementation.

Emotion names (live list): `amazed1 anxiety1 attentive1 attentive2 boredom1 boredom2 calming1 cheerful1 come1 confused1 contempt1 curious1 dance1 dance2 dance3 disgusted1 displeased1 displeased2 downcast1 dying1 electric1 enthusiastic1 enthusiastic2 exhausted1 fear1 frustrated1 furious1 go_away1 grateful1 helpful1 helpful2 impatient1 impatient2 incomprehensible2 indifferent1 inquiring1 inquiring2 inquiring3 irritated1 irritated2 laughing1 laughing2 lonely1 lost1 loving1 mini-deep-sleep no1 no_excited1 no_sad1 oops1 oops2 proud1 proud2 proud3 rage1 relief1 relief2 reprimand1 reprimand2 reprimand3 resigned1 sad1 sad2 scared1 serenity1 shy1 sleep1 success1 success2 surprised1 surprised2 thoughtful1 thoughtful2 tired1 toc-toc-toc uncertain1 uncomfortable1 understanding1 understanding2 waiting wake-mini-up welcoming1 welcoming2 yes1 yes_sad1` (regenerate with `emotions.list_moves()`).

Dance names (live list): `chicken_peck chin_lead dizzy_spin grid_snap groovy_sway_and_roll head_tilt_roll interwoven_spirals jackson_square neck_recoil pendulum_swing polyrhythm_combo sharp_side_tilt side_glance_flick side_peekaboo side_to_side_sway simple_nod stumble_and_recover uh_huh_tilt yeah_nod`.

## Recording your own motion

```python
mini.start_recording()
# ... drive the robot with set_target(...) calls ...
frames = mini.stop_recording()   # list of {"time", "head" (4x4 list), "antennas", "body_yaw"}
```

⚠️ Only `set_target` calls are recorded (they carry the record data); `goto_target` moves run on the daemon and
produce **nothing** (`stop_recording()` returns `[]` → indexing it raises) **(live)**.
To capture poses moved by hand: `enable_gravity_compensation()` and sample `get_current_head_pose()` in a loop.

## Motors

| Call | Effect |
|---|---|
| `enable_motors(ids=None)` | torque on (all or listed motors); pins targets to present pose |
| `disable_motors(ids=None)` | torque off — head falls if not supported |
| `enable_gravity_compensation()` / `disable_gravity_compensation()` | head movable by hand, holds against gravity |

REST equivalent: `POST /api/motors/set_mode/{enabled|disabled|gravity_compensation}`.

⚠️ Gravity compensation **only works with the Placo kinematics engine**. Our robot runs `AnalyticalKinematics`
(`GET /api/kinematics/info`), so the daemon raises `RuntimeError("Gravity compensation mode is only supported with
the Placo kinematics engine.")` and the SDK call silently does nothing useful. Changing the engine means restarting
the daemon with `--kinematics-engine Placo` (needs user confirmation). `make_motors_compliant()` seen in newer online
docs does **not** exist in 1.10.0.
`wake_up()` does not enable torque — enable motors first (preflight does it).

## Media (`mini.media`, a `MediaManager`)

Over the network the backend is **WebRTC** (needs the local GStreamer stack in the venv; the macOS
`libgstpython` warning is harmless). Wait ~1–2 s after connecting for the first frame.

### Camera

| Call | Notes |
|---|---|
| `media.get_frame()` | BGR `uint8` `(720, 1280, 3)`, **read-only** (`.copy()` before drawing) or `None` **(live)** |
| `media.get_frame_jpeg()` | ⚠️ returns empty over WebRTC **(live)**; encode yourself: `cv2.imencode(".jpg", frame)` |
| `media.camera.resolution` | `(1280, 720)` |
| `media.camera.K`, `media.camera.D` | intrinsics / distortion (also `GET /api/camera/specs`) |
| `mini.T_head_cam` | 4×4 head→camera transform |

Camera on this unit is blurry: autofocus motor (dw9807) fails over I2C — hardware, not fixable in code.

### Audio

| Call | Notes |
|---|---|
| `media.start_recording()` / `media.get_audio_sample()` / `media.stop_recording()` | float32 chunks `(320, 2)` at 16 kHz stereo **(live)**; poll `get_audio_sample()` often, `None` when no data |
| `media.get_input_audio_samplerate()`, `get_input_channels()` | 16000, 2 **(live)** |
| `media.start_playing()` / `media.push_audio_sample(np.float32 array)` / `media.stop_playing()` | stream audio to the speaker; mono is duplicated to the output channels |
| `media.get_output_audio_samplerate()`, `get_output_channels()` | 16000, 2 **(live)** |
| `media.play_sound(file)` | plays a file known to the daemon (`wake_up.wav`, `go_sleep.wav`, `count.wav`, `dance1.wav`, `confused1.wav`, `impatient1.wav`, or uploaded via REST) |
| `media.get_DoA()` | ⚠️ `None` over WebRTC **(live)** → use REST `GET /api/state/doa` (`angle` rad: 0 left, π/2 front, π right; `speech_detected`) |

`ERROR ... No Reachy Mini Audio USB device found!` on the Mac is harmless (looks for the USB audio of the Lite version).

### Releasing media (code running on the robot only)

`release_media()` / `acquire_media()` hand the camera/mic to direct OpenCV/sounddevice access and back.
From the Mac never release: daemon tracking and WebRTC stop getting frames.

## Head tracking & wobbling (daemon side)

```python
mini.start_head_tracking(weight=1.0)   # 1 = tracking owns head orientation; 0 = pause detection (cheap off)
face = mini.get_tracked_face(wait=False)
if face.detected:
    face.x, face.y   # nose position normalized to [-1, 1] on the image (x right, y down)
    face.roll        # rad
face.ts             # timestamp of the last processed frame; None → tracker gets no frames
mini.stop_head_tracking()
```

- Detection is YuNet on the robot; no face size/bbox is exposed.
- Pixel conversion: `cx = (x + 1) / 2 * (w - 1)`, `cy = (y + 1) / 2 * (h - 1)`.
- Intermediate weights blend tracking with your own `set_target` motion.
- `enable_wobbling()` / `disable_wobbling()`: head moves with audio being played (daemon sounds, WebRTC audio).

## Apps

A Reachy Mini app is a Python package exposing a `ReachyMiniApp` subclass; the daemon (dashboard / Control app)
installs it from a Hugging Face Space and runs it **on the robot**.

```python
import threading
from reachy_mini import ReachyMini, ReachyMiniApp

class MyApp(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"  # optional settings web page (serves static/index.html)
    request_media_backend: str | None = None            # e.g. "no_media"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            ...
```

- `wrapped_run()` builds the `ReachyMini` (localhost if the daemon is local, else network) and calls `run`.
- Stop via `stop()` (sets `stop_event`) — always poll `stop_event`.
- CLI (installed in the venv):
  - `reachy-mini-app-assistant create [--template default|conversation] [--publish] [--private] [app_name] [path]`
  - `reachy-mini-app-assistant check [app_path]`
  - `reachy-mini-app-assistant publish [--official] [--nocheck] [--private|--public] [app_path] [commit_message]`
- Lifecycle: the daemon runs the app as a subprocess `python -u -m <app>.main`; stop = SIGINT → `stop()` → `stop_event`;
  afterwards the daemon returns the robot to its default pose. Only one app at a time.
- On the wireless robot apps live in the shared venv `/venvs/apps_venv/` (daemon: `/venvs/mini_daemon/`).
- Generated project: `pyproject.toml`, `README.md` (frontmatter `tags: [reachy_mini_python_app]`), `index.html`,
  `style.css`, `<app>/{__init__.py, main.py, static/}`. Entry point:

  ```toml
  [project.entry-points."reachy_mini_apps"]
  my_app = "my_app.main:MyApp"
  ```

- `main.py` must end with:

  ```python
  if __name__ == "__main__":
      app = MyApp()
      try:
          app.wrapped_run()
      except KeyboardInterrupt:
          app.stop()
  ```

- With `custom_app_url`, add endpoints on `self.settings_app` (FastAPI) inside `run`; UI at `http://<robot>:8042`.
- Test locally: `python -m my_app.main` while a daemon runs (robot or `--sim`).
- Publish: `hf auth login` (write token) then `reachy-mini-app-assistant publish <path>`; app store: https://hf.co/reachy-mini/#/apps
- Install/start on the robot via REST (`/api/apps/install` body `{"url": "https://huggingface.co/spaces/<user>/<app>"}`,
  `/api/apps/start-app/<name>`) — installing/removing needs user confirmation.
- Offline install (confirmation needed): `scp -r my_app pollen@<robot>:/tmp/my_app` then
  `ssh reachy "/venvs/apps_venv/bin/pip install /tmp/my_app"`.
- Silent app crash → usually missing dependency: `ssh reachy "/venvs/apps_venv/bin/python3 -c 'from my_app.main import MyApp'"`.
- Conversation template (`--template conversation`) and the official app
  https://github.com/pollen-robotics/reachy_mini_conversation_app (HF realtime backend by default; OpenAI/Gemini optional).

## Simulation (no robot)

```bash
reachy_mini_env/bin/python -m pip install "reachy-mini[mujoco]"   # once
reachy_mini_env/bin/reachy-mini-daemon --sim            # MuJoCo window
reachy_mini_env/bin/mjpython -m reachy_mini.daemon.app.main --sim   # macOS: MuJoCo GUI needs mjpython
reachy_mini_env/bin/reachy-mini-daemon --sim --scene minimal        # scenes: empty (default) | minimal (table + objects)
reachy_mini_env/bin/reachy-mini-daemon --mockup-sim     # no MuJoCo, kinematics only
reachy_mini_env/bin/reachy-mini-daemon --sim --headless --no-media --fastapi-port 8000
```

The simulated robot behaves like a Lite on `localhost` (dashboard at http://127.0.0.1:8000/).
"Circular buffer overrun" warnings in sim = you are not consuming video → use `media_backend="no_media"`.
Then connect with `ReachyMini(connection_mode="localhost_only")`. `connect_robot()` targets the real robot;
for sim use `REACHY_HOST=localhost` (preflight works against any daemon).
Other daemon flags: `--scene`, `--check-collision`, `--kinematics-engine {Placo,NN,AnalyticalKinematics}`,
`--no-wake-up-on-start`, `--no-goto-sleep-on-stop`, `--preload-datasets`, `--log-level`.

## Hardware & limits (docs, wireless)

| Axis | Range |
|---|---|
| Head x / y / z translation | -1.5..+2.5 cm / ±4 cm / -4..+2.5 cm |
| Head roll / pitch | ±40° |
| Head yaw relative to body | ±60° (software: \|head yaw − body yaw\| ≤ 65°) |
| Body yaw | ±155° mechanical (docs quote ±160°/±180° software) |
| Antennas | ±180° |

The SDK/daemon clamps out-of-range targets. Stay well inside these ranges in demos.

- Head: Stewart platform (6× XL330-M288-T), body XC330-M288-PG, antennas 2× XL330-M077-T (IDs 10–18, 1 Mbps).
- Camera: Raspberry Pi Camera Module 3 **Wide** (IMX708, 120°, autofocus — broken on this unit).
- Audio: XVF3800 4-mic array (16 kHz, hardware AEC), 5 W speaker. DoA 0 rad = left, π/2 = front, π = right.
- IMU on wireless only. Battery LiFePO4 2000 mAh — charge level is **not** readable (LED only). When it dies the
  robot drops off Wi-Fi mid-command (moves stop, preflight exit 2); plug the power cable for long sessions.
- Compute: Raspberry Pi CM4 (4 GB). No USB data link on wireless: Wi-Fi only.
- Official docs: https://huggingface.co/docs/reachy_mini/index · repo https://github.com/pollen-robotics/reachy_mini
  (see `examples/`, `AGENTS.md`, `skills/`) · troubleshooting https://huggingface.co/docs/reachy_mini/troubleshooting

## Kinematics & utilities

- Engines: `AnalyticalKinematics` (default, ours — `GET /api/kinematics/info`), `Placo` (needed for gravity compensation), `NN`.
- URDF: `GET /api/kinematics/urdf`; STL meshes `GET /api/kinematics/stl/{filename}`.
- `reachy_mini.utils.interpolation`: `minimum_jerk(start, goal, duration)`, `linear_pose_interpolation(p0, p1, alpha)`,
  `time_trajectory(t_norm, method)`, `distance_between_poses(p1, p2) -> (m, rad, "magic-mm")`,
  `compose_world_offset(T_abs, T_offset, reorthonormalize=False)`.
- `reachy_mini.vision.look_at`: `look_at_world_pose(x, y, z)`, `look_at_image_pose(u, v, K, D, T_world_head, T_head_cam)`.

## Pitfalls checklist

1. Robot asleep / motors disabled → nothing moves. Preflight or `connect_robot()` first.
2. `no_media` releases the daemon camera → tracking `ts=None`. Use `connect_robot(media=False)`.
3. Frames are read-only → `.copy()`.
4. `goto_target` resets body yaw to 0 unless `body_yaw=None`.
5. `goto_target` blocks; for reactive control use `set_target` loops (or threads).
6. `get_frame_jpeg()` and `get_DoA()` don't work over WebRTC.
7. `start_recording` records only `set_target` calls.
8. `wake_up()`/`goto_sleep()`/`play_move(sound=True)` print "Audio system is not initialized" with `no_media` — harmless, just no sound.
9. `FaceTarget` has no size; x/y are the nose.
10. `goto_sleep()` leaves motors **enabled**.
11. Gravity compensation fails on our robot (AnalyticalKinematics engine).
12. Online docs track `main`; trust the installed 1.10.0 source when they disagree.
