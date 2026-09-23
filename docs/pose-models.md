# Pose models

Four pose backends behind one interface, and the measurements that decide which runs where.
See [simulation.md](simulation.md) to run them on the webcam and [robot.md](robot.md) for the robot.

## Body tracking and model comparison

Same idea as `head_tracking.py`, but the robot follows the **center of the torso** and the pose model is
selectable, so models can be compared under identical conditions (same loop, target rule and controller).

```bash
sim/start.sh                                                         # terminal 1
reachy_mini_env/bin/python -m sim.body_tracking --model blazepose-lite   # terminal 2
reachy_mini_env/bin/python -m sim.body_tracking --model blazepose-full
reachy_mini_env/bin/python -m sim.body_tracking --model vitpose-s
```

All `head_tracking` options (`--camera`, `--max-yaw`, `--smoothing`, ...) are available, plus `--min-score`.

| Model | Type | What runs | Keypoints |
|---|---|---|---|
| `blazepose-lite` / `blazepose-full` | single-stage (Google BlazePose, MediaPipe) | pose landmarker (internal person detection + tracking between frames) | 33 → mapped to COCO 17 |
| `vitpose-s` | top-down (ViTPose-S, transformer) | EfficientDet-Lite0 person detector **every frame**, then ViTPose on the 192×256 person crop (ONNX Runtime, CPU) | COCO 17 |

ViTPose pre/post-processing follows [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) (RGB, ImageNet
normalization, 3:4 crop, UDP heatmap decoding). On the same webcam frame its nose/eyes/shoulders land within
~0.04 (normalized) of BlazePose.

**Target rule:** mean of confident shoulders and hips (`torso`); if the hips are out of frame, the shoulders;
else the nose; else `lost` and the head drifts back to center. Confidence threshold per model
(`--min-score`, default 0.5 for BlazePose visibility, 0.3 for ViTPose heatmap peaks).

**Panel:** model and loop FPS; mean ms per stage over the last second (`detector`, `pose`, `total` = inference
only); confident keypoints; target source; yaw/pitch. On exit (`q`/`ESC`/`Ctrl+C`) a summary with mean and p95
per stage is printed.

Results on this Mac (Apple M1 Pro, CPU only, 1920×1080 frames, ~15 s runs with a person in view; measured
with the iPhone Continuity Camera by mistake — re-measure with `--camera FaceTime`) —
**not** Raspberry Pi numbers; measure on the Pi before choosing a model for the robot:

| Model | total mean (ms) | total p95 (ms) | loop FPS |
|---|---|---|---|
| blazepose-lite | 12.1 | 13.1 | 20.2 |
| blazepose-full | 18.7 | 19.7 | 15.6 |
| vitpose-s | 91.4 (detector 44.5 + pose 46.9) | 102.2 | 5.7 |

Loop FPS also includes camera capture, drawing and the display window, so it is lower than `1000 / total`.
ViTPose's cost is roughly half detector, half pose; running the detector only every few frames would be the
first optimization to try.

