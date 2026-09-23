# Activity tracking with the Reachy Mini robot

**Minor: Artificial Intelligence — Zuyd Hogeschool, 2026, blokperiode 1**
Commissioned by the Research Centre for Data Intelligence.
Client: Maarten Vaessen, Researcher Data Intelligence.

## Why this project exists

At the Research Centre for Data Intelligence, researchers and students carry out applied research across
five themes: Process Optimization, Digital Twins, Data Security and Infrastructure, Smart Assistants, and
Human Computer Interaction.

Within Smart Assistants, the research project **Measuring at Home with AI** aims to help children who
suffer from **JDM** (juvenile dermatomyositis), a rare muscle disease that causes muscle inflammation and
weakness. Because the disease is rare, these children are treated at the Wilhelmina Kinderziekenhuis (WKZ)
in Utrecht. Part of the treatment is monitoring how the disease develops through a series of exercises
(**CMAS**), which today must be performed in the presence of a specialized doctor from WKZ.

That requirement is the problem. Children and their parents have to travel to the hospital regularly just
to perform these exercises, and that has a large impact on their lives. The research centre's aim is to
reduce the number of visits by using AI-supported automatic scoring: the child performs the exercises at
home, in front of a monitoring system that scores them automatically.

## Project goal

This project explores what such a monitoring system could be. A smartphone or tablet would work, but a
device like the Reachy Mini may have an advantage beyond the camera: it can be perceived as a training or
exercise **buddy** rather than as a measuring instrument — which matters when the person being measured is
a child exercising at home without a doctor present.

The goal is therefore to investigate **whether the Reachy Mini is capable of person detection and activity
monitoring**, with the robot's own expressive features — head movement, gaze following, antenna gestures —
as part of the system rather than decoration.

### Intended result

1. A brief literature review on automatic person and activity detection systems.
2. A working prototype running on the Reachy Mini robot.

This repository is the second of those.

## What the prototype does today

The robot finds a person, follows them with its head and body, and recognizes what they are doing,
answering each recognized action with a distinct antenna gesture. Six classes are trained — none, wave,
push-up, squat, clapping, jumping jacks — over 3-second windows of skeleton keypoints.

Everything runs on CPU, on the robot itself: pose estimation via MoveNet int8, and a random forest exported
to plain numpy so it needs no scikit-learn on the Raspberry Pi.

**On what is and is not demonstrable, this repository tries to be blunt rather than flattering**, because a
scoring system that is trusted when it should not be is worse than none. Waving, clapping and jumping jacks
work. Squats miss roughly one repetition in seven with the legs in frame. Push-ups do not work, and are
unmeasured rather than proven broken. Held-out accuracy on public datasets also overstates real-room
performance for some classes — the honest figure for waving is 0.581, not the 0.918 the validation reports.

Measurements, caveats and the reasoning behind each number:
[docs/action-recognition.md](docs/action-recognition.md).

### Relevance to CMAS

The actions above are **not** CMAS exercises. They are public-dataset stand-ins chosen to test whether the
robot can do this class of task at all — the feasibility question the project asks. They exercise the same
pipeline a CMAS scorer would need: person detection, skeleton extraction on a constrained CPU, temporal
windowing, and per-repetition recognition with an honest confidence floor.

## Documentation

| Document | Covers |
|---|---|
| [docs/overview.md](docs/overview.md) | Technical overview: what runs where, quickstart, hardware constraints |
| [AGENTS.md](AGENTS.md) | Operating manual: robot host, preflight, autonomy rules, known pitfalls |
| [docs/simulation.md](docs/simulation.md) | MuJoCo simulation and webcam tracking, with no robot |
| [docs/robot.md](docs/robot.md) | Going from simulation to the physical robot |
| [docs/pose-models.md](docs/pose-models.md) | The four pose backends and their measured cost |
| [docs/action-recognition.md](docs/action-recognition.md) | Activity recognition, training, and the honest per-class results |
| [docs/demo.md](docs/demo.md) | The one-command demo and how it fails |
| [datasets/SOURCES.md](datasets/SOURCES.md) | Dataset provenance, checksums and licences |

## Resources used

- Reachy Mini robot (wireless, Raspberry Pi CM4).
- Open-source human pose detection models: MoveNet, BlazePose, ViTPose-S.
- Public skeleton datasets for training: NTU RGB+D 60, UCF101, HMDB51 — see
  [datasets/SOURCES.md](datasets/SOURCES.md) for licences, which are research-use only.

## Licence

MIT — see [LICENSE](LICENSE).
