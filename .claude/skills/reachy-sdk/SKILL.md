---
name: reachy-sdk
description: Reachy Mini Python SDK (reachy_mini 1.10.0) guide. Use when writing, reviewing or debugging Python code for the robot - head/antenna/body motion, goto_target/set_target, emotions and dances, recording moves, camera frames, audio capture/playback, face tracking, motors/torque, IMU, Reachy Mini apps, simulation or kinematics.
---

# Reachy Mini Python SDK

Full reference with signatures, units and live-verified behaviour: [reference.md](reference.md).
Working examples: `examples/` (run with `reachy_mini_env/bin/python -m examples.<name>`).

## Mandatory pattern in this project

```python
from robot.connect import connect_robot

with connect_robot(media=False, needs=("motion",), sleep_on_exit=False) as mini:
    ...
```

`connect_robot()` runs the preflight with auto-fix (reachable, daemon, backend, motors enabled, awake, media),
then opens `ReachyMini(host, connection_mode="network", ...)`. Never construct `ReachyMini` directly unless you
have a reason (e.g. code running on the robot, simulation) — and then run the preflight yourself.

| Script needs | Use |
|---|---|
| Only motion / state | `connect_robot(media=False)` |
| Daemon face tracking, no video here | `connect_robot(media=False, needs=("tracking",))` |
| Video frames / audio on this Mac | `connect_robot(media=True)` (WebRTC) |
| Video + tracking | `connect_robot(media=True, needs=("tracking",))` |
| Leave robot asleep at the end | add `sleep_on_exit=True` |

Head tracking is stopped automatically on exit. Keep loops bounded or catch `KeyboardInterrupt`.

## Cheat sheet

```python
import time
import numpy as np
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMoves

# Poses: 4x4 matrices. x fwd, y left, z up. Angles in degrees by default, mm=True for millimeters.
mini.goto_target(head=create_head_pose(pitch=-10, yaw=20), antennas=[0.5, -0.5], duration=1.0)  # blocking
mini.goto_target(head=create_head_pose(z=10, mm=True), duration=0.5, method="cartoon")
mini.goto_target(body_yaw=np.deg2rad(30), duration=1.0)           # body only (radians)
mini.goto_target(head=create_head_pose(yaw=20), body_yaw=None)    # keep current body yaw!
mini.set_target(head=create_head_pose(roll=5))                    # immediate; loop at ~50 Hz
mini.look_at_world(0.5, 0.2, 0.1, duration=1.0)                   # meters, robot frame
mini.look_at_image(640, 360, duration=0.5)                        # pixel on the camera image (media=True)
mini.wake_up(); mini.goto_sleep()

pose = mini.get_current_head_pose()                               # 4x4
head_joints, antennas = mini.get_current_joint_positions()
imu = mini.imu                                                    # accelerometer, gyroscope, quaternion, temperature

emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
mini.play_move(emotions.get("cheerful1"), initial_goto_duration=1.0, sound=True)

mini.start_head_tracking(weight=1.0)
face = mini.get_tracked_face(wait=False)   # detected, x, y in [-1,1] (nose), roll rad, ts

frame = mini.media.get_frame()             # BGR (720,1280,3) read-only → frame.copy()
mini.media.start_recording(); chunk = mini.media.get_audio_sample()   # float32 (320, 2) @ 16 kHz
mini.media.play_sound("count.wav")
```

## Pitfalls (most common first)

1. Robot asleep / motors disabled → commands accepted but nothing moves. Use `connect_robot()` / preflight.
2. `media_backend="no_media"` releases the daemon camera → tracker `ts=None`. `connect_robot(media=False)` fixes it.
3. SDK frames are read-only → `.copy()` before OpenCV drawing.
4. `goto_target(...)` resets body yaw to 0 unless `body_yaw=None`.
5. `goto_target` blocks for `duration`; use `set_target` loops or threads for reactive behaviour.
6. Over WebRTC (this Mac): `media.get_frame_jpeg()` is empty and `media.get_DoA()` is `None` → encode with OpenCV;
   DoA via REST `GET /api/state/doa`.
7. `start_recording()` captures only `set_target` calls, not `goto_target`.
8. "Audio system is not initialized" with `media=False` → sounds from `wake_up`/`play_move` are skipped (harmless).
9. macOS: GStreamer `libgstpython` warning and `No Reachy Mini Audio USB device found!` are harmless.
10. `FaceTarget` has no face size; the camera image is blurry on this unit (hardware focus failure).
11. Keep SDK and daemon versions equal (both 1.10.0 now); a mismatch warns and may break.
12. Gravity compensation needs the Placo kinematics engine — ours runs AnalyticalKinematics, so it fails.

## Where to look next

- Apps (`ReachyMiniApp`, `reachy-mini-app-assistant create|check|publish`), simulation (`reachy-mini-daemon --sim`),
  kinematics, interpolation helpers, custom `Move` classes → [reference.md](reference.md).
- REST equivalents / things the SDK does not expose (volume, apps, sounds upload, camera specs) → `reachy-api` skill.
- Robot-side logs and diagnostics → `reachy-ssh` skill.
