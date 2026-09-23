---
name: reachy-api
description: Reachy Mini daemon REST API reference (port 8000). Use when calling the robot over HTTP/curl, checking daemon/motor/media/tracking status, moving the robot via API, playing emotions/dances or sounds, managing apps, volume, camera specs, kinematics, Wi-Fi, Hugging Face auth or updates, streaming daemon logs, or reading the Swagger/openapi docs.
---

# Reachy Mini daemon REST API

The daemon runs on the robot (FastAPI, `reachy_mini` 1.10.0) and is what both the SDK and the
Reachy Mini Control app talk to.

- Base URL: `http://<host>:8000` — host `192.168.137.171` (**may change**), fallback `reachy-mini.local`.
- Robot and computer **must be on the same Wi-Fi** (currently `Reachy Mini`).
- Swagger UI: `http://<host>:8000/docs` · raw spec: `/openapi.json`.
- Full per-endpoint reference (params, bodies, schemas, live example responses): [endpoints.md](endpoints.md).
  Regenerate it after a daemon update: `reachy_mini_env/bin/python tools/gen_api_reference.py`.

## Before any call that moves or uses media

Run the preflight (it wakes the robot, enables motors, re-acquires the camera):

```bash
reachy_mini_env/bin/python robot/preflight.py --fix [--need media] [--need tracking]
```

`GET /api/daemon/status` alone is **not** enough: it does not tell you the robot is asleep or the camera hidden.

## Units and conventions

- Head pose `XYZRPYPose`: `x, y, z` in **meters**, `roll, pitch, yaw` in **radians** (all default 0).
  Alternative `Matrix4x4Pose`: `{"m": [16 floats, row-major]}` (translation in meters).
  Positive pitch = head looks **down**; the sleep pose is `z≈-0.045, pitch≈+0.48`.
- Antennas: `[right, left]` from the robot's point of view, radians (verified visually; the openapi docstring
  "(left, right)" is wrong). Sleep ≈ `[-3.05, 3.05]`, neutral `[-0.1745, 0.1745]`.
- Body yaw: radians.
- `InterpolationTechnique`: `linear`, `minjerk` (default), `ease_in_out`, `cartoon`.
- `MotorControlMode`: `enabled`, `disabled`, `gravity_compensation`.
- Moves return `{"uuid": "..."}`; `GET /api/move/running` lists running move UUIDs.
- Reached pose can differ from the requested one (kinematic limits, automatic body yaw).
- Recorded move datasets are Hugging Face datasets; the name goes in the path **with its slash**:
  `pollen-robotics/reachy-mini-emotions-library`, `pollen-robotics/reachy-mini-dances-library`
  (both pre-downloaded on the robot at daemon start).

## Quick recipes (all tested live)

```bash
H=http://192.168.137.171:8000

# --- status / state ---
curl -s $H/api/daemon/status                       # state, version, backend ready, media_released, face_target
curl -s $H/api/state/full                          # control_mode, head_pose, body_yaw, antennas_position
curl -s "$H/api/state/full?with_doa=true&with_head_joints=true"
curl -s "$H/api/state/present_head_pose?use_pose_matrix=true"
curl -s $H/api/motors/status                       # {"mode": "enabled"}
curl -s $H/api/move/running                        # [] when idle
curl -s $H/api/media/status                        # {"available":true,"released":false,"no_media":false}
curl -s $H/api/state/doa                           # {"angle": rad (0=left, pi/2=front, pi=right), "speech_detected": bool}

# --- motors / sleep ---
curl -s -X POST $H/api/motors/set_mode/enabled
curl -s -X POST $H/api/motors/set_mode/gravity_compensation   # move the head by hand
curl -s -X POST $H/api/move/play/wake_up
curl -s -X POST $H/api/move/play/goto_sleep

# --- motion ---
curl -s -X POST $H/api/move/goto -H 'Content-Type: application/json' -d '{
  "head_pose": {"x":0,"y":0,"z":0,"roll":0,"pitch":0.17,"yaw":0.3},
  "antennas": [0.4,-0.4], "body_yaw": 0.0, "duration": 1.0, "interpolation": "minjerk"}'
# instant target (no interpolation; send repeatedly for streaming control)
curl -s -X POST $H/api/move/set_target -H 'Content-Type: application/json' -d '{
  "target_head_pose": {"x":0,"y":0,"z":0,"roll":0,"pitch":0,"yaw":0}, "target_antennas": [0,0]}'

# --- emotions & dances ---
curl -s $H/api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-emotions-library
curl -s $H/api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-dances-library
curl -s -X POST $H/api/move/play/recorded-move-dataset/pollen-robotics/reachy-mini-emotions-library/curious1
curl -s -X POST $H/api/move/stop -H 'Content-Type: application/json' -d '{"uuid":"<uuid>"}'

# --- camera / tracking ---
curl -s -X POST $H/api/media/acquire               # give camera+audio back to the daemon
curl -s -X POST $H/api/media/release               # hand camera+audio to a client (daemon tracker stops getting frames)
curl -s -X POST $H/api/media/tracking/enable -H 'Content-Type: application/json' -d '{"weight":1.0}'  # 0 = pause detection
curl -s $H/api/media/tracking/face                 # face_target: detected, x/y in [-1,1] (nose), roll rad, ts
curl -s -X POST $H/api/media/tracking/disable
curl -s $H/api/camera/specs                        # resolutions, intrinsics K, distortion D

# --- sound / volume ---
curl -s -X POST $H/api/media/play_sound -H 'Content-Type: application/json' -d '{"file":"count.wav"}'
#   built-in files: wake_up.wav go_sleep.wav count.wav dance1.wav confused1.wav impatient1.wav
curl -s -X POST $H/api/media/sounds/upload -F 'file=@hello.wav'   # -> /tmp/reachy_mini_sounds/hello.wav, then play_sound {"file":"hello.wav"}
curl -s $H/api/media/sounds
curl -s -X POST $H/api/media/stop_sound
curl -s -X POST $H/api/media/wobbling/enable       # head wobbles with played audio
curl -s $H/api/volume/current                      # {"volume": 0-100, ...}
curl -s -X POST $H/api/volume/set -H 'Content-Type: application/json' -d '{"volume":60}'   # plays a test sound
curl -s -X POST $H/api/volume/microphone/set -H 'Content-Type: application/json' -d '{"volume":100}'

# --- apps (Hugging Face Spaces) ---
curl -s $H/api/apps/list-available/installed       # source_kind: see SourceKind in endpoints.md
curl -s $H/api/apps/current-app-status             # null when idle
curl -s $H/api/daemon/robot-app-lock-status        # {"state":"free","holder_name":null}
curl -s -X POST $H/api/apps/start-app/<app_name>
curl -s $H/api/apps/job-status/<job_id>            # install/remove/update jobs

# --- misc ---
curl -s $H/api/kinematics/info
curl -s $H/wifi/status                             # mode, known_networks, connected_network
curl -s $H/update/available
```

## WebSockets (not in openapi.json)

| URL | Purpose |
|---|---|
| `ws://<host>:8000/ws/sdk` | SDK control channel (used by `ReachyMini`; don't talk to it by hand) |
| `ws://<host>:8000/logs/ws/daemon` | Live `journalctl -u reachy-mini-daemon -b -f` (last 100 lines first; empty strings are keepalives) |
| `ws://<host>:8000/api/apps/ws/apps-manager/{job_id}` | Progress of an app install/remove/update job |
| `ws://<host>:8000/update/ws/logs?job_id=<id>` | Progress of a daemon update job |

Read logs from the Mac with a bounded loop (macOS has no `timeout`):

```bash
reachy_mini_env/bin/python - <<'EOF'
import time, websockets.sync.client as ws
with ws.connect("ws://192.168.137.171:8000/logs/ws/daemon") as c:
    end = time.time() + 5
    while time.time() < end:
        try:
            line = c.recv(timeout=1)
            if line: print(line)
        except TimeoutError:
            pass
EOF
```

## 🔴 Dangerous endpoints — ask the user first

- Daemon: `POST /api/daemon/start|stop|restart`, `POST /api/daemon/robot-name`
- Updates: `POST /update/start`, `POST /update/start-from-ref`
- Wi-Fi: every `POST /wifi/*` (can disconnect the robot from this network → unreachable)
- Cache: `POST /cache/clear-hf`, `POST /cache/reset-apps`
- Apps: `install`, `install-private-space`, `remove`, `update`, `stop-current-app`, `restart-current-app`, `PUT startup-app`
- Audio DSP: `POST /api/audio/config/apply` (XVF3800 parameters)
- HF auth: `save-token`, `DELETE token`, `refresh-relay`

`POST /health-check` resets a watchdog timer used by clients; don't call it casually.

## Common errors

- `422 Unprocessable Entity` → body/param shape wrong; check the schema in endpoints.md.
- Move accepted but nothing happens → motors disabled or robot asleep → run the preflight.
- `tracking/face` always `ts: null` → camera released by a client (`no_media` SDK connection) → `POST /api/media/acquire`.
- Connection refused/timeout → robot off, other Wi-Fi, or IP changed (preflight exit 2).
