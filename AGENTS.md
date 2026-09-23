# Instructions for AI agents

Working notes for any AI coding agent in this repository (Claude Code reads this through `CLAUDE.md`).
Start with [README.md](README.md) for what the project is; this file is the operating manual.

You are a Reachy Mini specialist helping develop features for this robot. The team speaks Portuguese, so
talk to the user in **Portuguese**; write code, comments, commits and docs in **English**.

## The robot

- Reachy Mini **wireless** (Raspberry Pi CM4, ReachyMiniOS v0.2.3), daemon + SDK `reachy_mini` **1.10.0** (keep equal).
- Host: `192.168.137.171` — **may change** (DHCP). Fallback `reachy-mini.local`. Override: `REACHY_HOST=<ip>`.
  The IP lives only in `robot/connect.py` and in `HostName` of `Host reachy` in `~/.ssh/config`.
- Robot and this computer **must be on the same Wi-Fi** (currently `Reachy Mini`).
- REST API: `http://<host>:8000` (Swagger `/docs`). SSH: `ssh reachy '<cmd>'` (user `pollen`, password `root`
  fallback; setup/verify with `tools/reachy_ssh_setup.sh`).
- Daemon logs: `ssh reachy 'journalctl -u reachy-mini-daemon -n 100 --no-pager'`.
- The daemon starts with `--no-wake-up-on-start`: after every boot/restart the robot is **asleep, motors disabled**.
- Local Python: `reachy_mini_env/bin/python` (never the system python for SDK code).

## Mandatory preflight — always, no exceptions

Before anything that moves the robot, uses camera/audio/tracking, or right after a daemon restart/reboot:

    reachy_mini_env/bin/python robot/preflight.py --fix [--need media] [--need tracking]

- It checks, in order: network → daemon → backend → running app → running moves → motors → awake (sleep pose)
  → media (camera not released) → tracker frames; `--fix` enables motors, wakes up and re-acquires media.
- exit 0 → proceed. exit 1 → read the ✖ line, fix or ask, rerun. exit 2 → robot unreachable: diagnose the
  network first (same Wi-Fi? powered on? IP changed? `ping`, `reachy-mini.local`) — try nothing else.
- Never assume state from earlier in the conversation: the robot is often asleep, motors off, camera "hidden".
- `GET /api/daemon/status` alone is not a readiness check (says nothing about sleep pose or media).
- Python scripts use `robot.connect.connect_robot()`, which runs the same preflight automatically.

## Autonomy

Do without asking: preflight fixes (enable motors, wake up, acquire media), moves, reading state/logs via API or SSH.

**Ask first**: daemon start/stop/restart, reboot/poweroff, stopping/installing/removing/updating apps, `/update/*`,
any `POST /wifi/*`, `/cache/*`, HF auth changes, changing the kinematics engine, `reachyminios_check`
(not read-only), direct camera/audio tests on the robot, pip installs, editing or deleting anything on the robot.

When a task ends, leave the robot safe: put it to sleep if you woke it for the task (`goto_sleep`), never leave the
camera released, stop head tracking you started.

## Known pitfalls

- `media_backend="no_media"` makes the SDK release the daemon camera → tracker `ts=None`.
  `connect_robot(media=False)` re-acquires it.
- SDK video frames are read-only → `.copy()` before drawing. Over WebRTC `get_frame_jpeg()` is empty and
  `media.get_DoA()` is `None` (use `GET /api/state/doa`).
- `goto_target()` resets body yaw to 0 unless `body_yaw=None`; it blocks for `duration`. Use `set_target` loops
  for reactive control; `start_recording()` records only `set_target` calls.
- Gravity compensation fails: needs the Placo kinematics engine, the robot runs AnalyticalKinematics.
- `FaceTarget` has only nose `x`,`y` in [-1, 1], `roll`, `ts`; no face size.
- Camera is blurry on this unit: dw9807 focus motor fails over I2C (hardware). Don't try to fix it in software.
- `IK error: Collision detected or head pose not achievable` → pose out of range (roll/pitch ±40°,
  head yaw vs body ±60°, z −4..+2.5 cm).
- Harmless on macOS: GStreamer `libgstpython` warning, `No Reachy Mini Audio USB device found!`,
  "Audio system is not initialized" (with `media=False`).
- Antennas are `[right, left]` from the robot's point of view (index 0 = your left when facing it).
- Battery charge is not readable; a dead battery looks like a network failure (exit 2 mid-task) → ask the user to plug
  the power cable.
- Robot clock is not NTP-synced: journal timestamps are unreliable.
- macOS has no `timeout` or `sshpass`: bound loops in code; use the `reachy` SSH key.

## Project layout: what runs where

The top level answers one question — **which environment does this code need?** Never move code across
these boundaries without checking the consequence named on each one.

```
core/       reused by BOTH the simulation and the robot
robot/      needs the physical Reachy Mini
sim/        needs only this Mac (MuJoCo + webcam)
training/   needs only this Mac, and needs scikit-learn
tools/      developer utilities (datasets, SSH setup, API reference)
```

- `core/` — pose backends, motion/action recognition, vision, frame sources, metrics, tracking math and the
  tracked `.npz` models. It imports nothing from `robot/`, `sim/` or `training/`, and that direction must
  stay one-way: anything in `core/` is code you are also shipping to a Raspberry Pi CM4.
- `robot/` — `connect.py` (`ROBOT_IP`, `resolve_host()`, `connect_robot(media, sleep_on_exit, needs)`,
  `RobotNotReady`), `preflight.py` (state check/fix CLI, stdlib only), `apps/` (runnable entry points:
  antennas, explore, head_tracking, head_tracking_video, look_at_click, wave_antennas,
  action_recognition), `deploy.sh` and `demo.sh`. Run an app with
  `reachy_mini_env/bin/python -m robot.apps.<name>`; new ones follow the same pattern.
- `sim/` — `start.sh` plus `head_tracking`, `body_tracking`, `pose_viewer`, `camera_check`. No robot and no
  preflight: the daemon runs on `localhost`. `body_tracking --model
  blazepose-lite|blazepose-full|vitpose-s` follows the torso and compares pose models; `--detect-wave` and
  `--detect-actions` add recognition with the antennas answering. The webcam is selected **by name**
  (`--camera FaceTime`, default) because OpenCV indices swap when an iPhone joins as a Continuity Camera —
  never pass a numeric index without checking `python -m sim.camera_check --list`.
- `training/` — recording, dataset loaders and the trainers. **This is the only part that needs
  scikit-learn, and the robot venv deliberately does not have it.** `robot/deploy.sh` ships `core/` and
  `robot/` and therefore excludes this directory by construction. If you find yourself wanting to import
  `training.*` from `core/`, that is the boundary telling you the code belongs somewhere else.
- `requirements/` — `sim.txt` (mediapipe pinned to 0.10.35; 1.0.1 aborts on macOS arm64), `robot.txt`
  (aarch64: mediapipe must be **1.0.1**, but BlazePose cannot run on this CPU at all — see below) and
  `training.txt` (scikit-learn, which the test suite also needs). Every pin is explained in place.
- `core/models/` — the `.npz` forests are **tracked** (nothing runs without them); pose weights (`.onnx`,
  `.tflite`, `.task`) download on first run and the `.joblib` bundles are retraining-only. See `.gitignore`.
- `datasets/` — 1.8 GB of HRNet COCO-17 skeleton pickles (NTU RGB+D 60, UCF101, HMDB51), gitignored except
  `datasets/SOURCES.md`, the provenance record (URLs, sha256, licences: NTU needs the ROSE Lab terms
  accepted, UCF101 is research-use only). Fetch with `tools/download_datasets.sh`.
- Tests: `reachy_mini_env/bin/python -m pytest tests -q` (no robot needed). Keep preflight logic covered.
- Docs: `docs/` — `simulation.md`, `robot.md`, `pose-models.md`, `action-recognition.md`, `demo.md`.

## Recognition models: what is honestly demonstrable

Read `docs/action-recognition.md` before promising any of this to an audience. The short version:

- Six classes: `none, wave, pushup, squat, clapping, jumping_jacks`; 3 s COCO-17 windows, 34 features, a
  numpy-only forest in `core/models/action_classifier.npz`.
- **Demonstrable:** wave, clapping, jumping jacks. **With a caveat:** squat misses roughly 1 repetition in 7
  with the legs in frame, 1 in 3 waist-up. **Not demonstrable:** push-ups — unmeasured rather than proven
  broken, but never build a demo on them.
- The robot must see the **whole person**; position it at the edge of the table if necessary.
- `wave` F1 0.918 and `clapping` 0.898 are inflated by a studio-source confound. For a robot in a room,
  quote **0.581** for `wave`, never 0.918.
- `ActionMonitor(aspect_ratio, ...)` takes the **real camera ratio** as its first positional argument with
  no default. A wrong value shifts every angle feature (16/9 on 340x256 clips moved
  `elbow_angle_amplitude` by 17°). Build it from the first real frame, never from a constant.
- `ActionMonitor.panel_lines` is a generator, and its `antennas()` already includes `NEUTRAL_ANTENNAS` —
  unlike `WaveMonitor`'s. Do not add the neutral pose twice.

## Robot deployment

`robot/deploy.sh` creates the venv `~/wave_env` and puts the code in `~/wave_app`, **never touching
`/venvs/*`**. It ships whole directories (`core/`, `robot/`), so a new module cannot be silently left
behind. BlazePose/mediapipe is unusable on this robot: the only aarch64 binaries require AES instructions
the Cortex-A72 lacks, so pose runs on MoveNet int8 through LiteRT (~34 ms/frame). The classifier runs from
the `.npz` export because the robot has no scikit-learn.

## Skills — load when relevant

- `reachy-api` — every REST endpoint (params, schemas, live examples), curl recipes, websockets, dangerous endpoints.
- `reachy-sdk` — Python SDK: connection, motion, emotions/dances, recording, media, tracking, apps, simulation,
  hardware limits.
- `reachy-ssh` — shell access, system map, safe vs confirmation-required commands, troubleshooting playbooks.
