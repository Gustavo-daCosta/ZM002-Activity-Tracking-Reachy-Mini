# ZM002 — Activity Tracking with Reachy Mini

Body tracking and human-activity recognition on a [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/)
wireless: the robot watches a person, follows them with its head and body, and recognizes what they are
doing — waving, clapping, squatting, jumping jacks — answering each with a distinct antenna gesture.

Everything runs on CPU. The recognizer is a 34-feature random forest over 3-second skeleton windows,
exported to plain numpy so it runs on the robot's Raspberry Pi CM4 without scikit-learn.

## What runs where

The top-level directories are the four execution environments. This is the repository's main organizing
idea, because the robot is a 2 GB Raspberry Pi that cannot run most of what a laptop can.

| Directory | Needs | Contents |
|---|---|---|
| `core/` | **both** | Pose backends, motion & action recognition, vision, frame sources, metrics, tracked `.npz` models |
| `robot/` | the physical robot | Connection, mandatory preflight, runnable apps, deployment, the demo script |
| `sim/` | only this computer | MuJoCo simulation + webcam tracking — no robot, no preflight |
| `training/` | only this computer | Recording, dataset loaders, trainers — **the only part needing scikit-learn** |

`core/` imports nothing from the other three, and that stays one-way: whatever lands in `core/` is also
being shipped to the robot. `robot/deploy.sh` copies `core/` and `robot/` wholesale, so `training/` is
excluded by construction rather than by a hand-maintained file list.

## Quickstart

Both paths use the repo virtualenv `reachy_mini_env` (Python 3.12, `reachy_mini==1.10.0`).

**No robot — simulation and webcam.** Nothing here can move real hardware.

```bash
reachy_mini_env/bin/python -m pip install -r requirements/sim.txt
./sim/start.sh                                    # MuJoCo daemon + 3D viewer on localhost:8000
reachy_mini_env/bin/python -m sim.body_tracking --detect-actions
```

**With the robot.** Every entry point runs a mandatory preflight first; see [AGENTS.md](AGENTS.md).

```bash
./robot/deploy.sh                                 # venv ~/wave_env, code in ~/wave_app
./robot/demo.sh                                   # one command: sync, preflight, run
```

Retraining a model, or running the test suite, also needs `requirements/training.txt` (scikit-learn):

```bash
reachy_mini_env/bin/python -m pip install -r requirements/training.txt
reachy_mini_env/bin/python -m pytest tests -q          # neither robot nor webcam needed
```

## What it recognizes, honestly

Six classes (`none`, `wave`, `pushup`, `squat`, `clapping`, `jumping_jacks`), validated on held-out
subjects. The numbers that matter before you demo anything:

| Class | Demonstrable? |
|---|---|
| wave, clapping, jumping jacks | **Yes** |
| squat | **With a caveat** — misses ~1 repetition in 7 with the legs in frame, ~1 in 3 waist-up |
| push-ups | **No** — unmeasured rather than proven broken; never build a demo on them |

Two caveats that are easy to quote wrongly:

- The robot must see the **whole person**. Position it at the edge of a table if necessary; waist-up
  framing costs real accuracy, because the leg keypoints carry the squat and jumping-jack signal.
- `wave` scores F1 0.918 on held-out dataset subjects, but that number is inflated by a studio-source
  confound. For a robot in an ordinary room, **0.581** is the honest figure.

Full per-class measurements, the training pipeline and the source confound:
[docs/action-recognition.md](docs/action-recognition.md).

## Documentation

| Document | Covers |
|---|---|
| [AGENTS.md](AGENTS.md) | Operating manual: robot host, preflight, autonomy rules, known pitfalls, layout |
| [docs/simulation.md](docs/simulation.md) | MuJoCo simulation and webcam tracking |
| [docs/robot.md](docs/robot.md) | Going from simulation to the physical robot |
| [docs/pose-models.md](docs/pose-models.md) | The four pose backends and their measured cost |
| [docs/action-recognition.md](docs/action-recognition.md) | Wave and six-class recognition, training, honest F1 |
| [docs/demo.md](docs/demo.md) | The one-command demo and how it fails |
| [datasets/SOURCES.md](datasets/SOURCES.md) | Dataset provenance, checksums and licences |

## Hardware notes

Reachy Mini wireless, Raspberry Pi CM4 (Cortex-A72), ReachyMiniOS v0.2.3, `reachy_mini` 1.10.0.

MediaPipe/BlazePose **cannot run on this robot**: the only aarch64 binaries require AES instructions the
Cortex-A72 lacks and abort on startup. Pose estimation on the robot is MoveNet Lightning int8 via LiteRT
(~34 ms/frame); BlazePose and ViTPose-S are laptop-only.

## Licence

MIT — see [LICENSE](LICENSE).
