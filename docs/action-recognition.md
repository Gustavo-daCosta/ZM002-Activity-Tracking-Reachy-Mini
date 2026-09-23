# Motion recognition

Hand-wave recognition and the six-class action recognizer: how they are trained, what they measure,
and what is honestly demonstrable. Read the F1 table before promising anything to an audience.

## Motion recognition: hand wave

On top of pose estimation, `body_tracking --detect-wave` recognizes a **hand wave** and the robot waves back with
its antennas. Two detectors run side by side on the same 1.5 s sliding window of poses (normalized to the shoulder
midpoint and shoulder width, with a frame aspect-ratio correction), so they can be compared live:

- **rules** (`motion/detectors.py`): wrist above the shoulder, ≥3 horizontal reversals, ≥0.3 shoulder-widths of
  x amplitude;
- **classifier** (`motion/train.py`): RandomForest on 9 window features (reversals, reversal rate, amplitudes,
  speed, height, validity), saved to `models/wave_classifier.joblib`.

```bash
reachy_mini_env/bin/python -m training.record          # guided recording (WAVE / NOT WAVE rounds)
reachy_mini_env/bin/python -m training.train           # trains + honest grouped evaluation
reachy_mini_env/bin/python -m sim.body_tracking --detect-wave [--wave-trigger rules|classifier]
```

Measured on the 2 recorded sessions (416 windows, 220 wave), evaluated leave-one-session-out:

| Detector | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| rules | 0.832 | 1.000 | 0.682 | 0.811 |
| classifier (trained on those sessions) | 0.880 | 0.905 | 0.864 | 0.884 |
| **classifier trained on NTU** (default) | **0.950** | **0.976** | 0.927 | **0.951** |

The default model is the NTU-trained one (see below): it never saw these recordings and still scores higher on
them than the model fitted to them. `--classifier core/models/wave_classifier.joblib` selects the
self-trained one.

The rules never fire on a non-wave but miss a third of the waves (the median window has exactly 3 reversals, the
threshold); the classifier trades a little precision for clearly better recall. Top feature importances:
`reversal_rate` 0.274, `reversals` 0.249.

### Training on public datasets

The two recorded sessions are one person, one camera, one room. `motion/public_data.py` reads the COCO-17
skeleton files published by MMAction2 / PySkl (no video download, no pose estimation to run) and turns them
into the same 1.5 s windows, resampled to the robot's ~10 FPS:

| file | size | content |
|---|---|---|
| [`ntu60_hrnet.pkl`](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl) | 705 MB | NTU RGB+D 60: 56 880 clips, 40 subjects, 3 cameras; class **22 = "A23 hand waving"**, the other 59 actions are labelled negatives |
| [`hmdb51_2d.pkl`](https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl) | 203 MB | HMDB51: 6 371 movie clips; class **50 = "wave"** (104 clips), in the wild - kept as an external test set |

```bash
reachy_mini_env/bin/python -m training.train_public --ntu <ntu60_hrnet.pkl> --hmdb <hmdb51_2d.pkl>
```

Evaluation is grouped by subject, so the reported numbers are for people the model never saw; HMDB51 and our
own recordings are only ever used as external test sets. The decision threshold is tuned on the held-out
predictions and saved with the model (0.75 for the shipped one).

Measured with 32 524 windows (2 986 wave) from 40 subjects, 120 clips per negative class:

| set | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| rules, same windows | 0.912 | 0.541 | 0.246 | 0.339 |
| classifier, unseen subjects | 0.897 | 0.460 | 0.703 | 0.556 |
| HMDB51 (movies), external | 0.905 | 0.040 | 0.188 | 0.065 |
| our robot recordings, external | 0.950 | 0.976 | 0.927 | 0.951 |

NTU is harder than our own recordings because it contains actions built to be confused with a wave (taking a
selfie, cheering up, brushing hair, salute). HMDB51 scores badly for a different reason: movie clips with
moving cameras, partial bodies and crowds where the tracked person is often not the one waving — it is a
pessimistic external check, not a target. Cite Shahroudy et al. (2016) and Liu et al. (2019)
for NTU, Kuehne et al. (2011) for HMDB51; both are for research use.

## Motion recognition: six actions

`--detect-actions` recognizes **`none`, `wave`, `pushup`, `squat`, `clapping`, `jumping_jacks`** from a 3 s
sliding window of COCO-17 keypoints, and the antennas answer each one with its own amplitude/frequency
signature, so you can tell from across the room what was recognized. One model (`core/motion/actions/`,
shipped as `models/action_classifier.npz`) runs on both the Mac and the robot: it is COCO-17 only, numpy
only, and needs no scikit-learn at runtime.

```bash
# Mac webcam. The camera is chosen by NAME, never by index (indices move when an iPhone joins
# as a Continuity Camera): see `python -m sim.camera_check --list`.
reachy_mini_env/bin/python -m sim.body_tracking --camera FaceTime --model blazepose-lite --detect-actions

# Robot (headless; deploy first with robot/deploy.sh)
ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.action_recognition --seconds 60'
```

### Positioning is load-bearing

**Place the robot so its camera sees the whole person** — at the edge of the table, or stand further back.
Two framings were measured: *full-body* (the whole person) and *waist-up* — hips, knees and ankles unseen,
which is what a desk-height camera gives (measured over 7 225 frames of our own robot recordings: knees and
ankles confident in 0.000 of frames, hips in 0.19). The framing is simulated by removing those keypoints'
confidence before windowing and normalization, so the whole pipeline sees a genuinely cropped skeleton. The
difference is not cosmetic: waist-up squats lose a third of their repetitions and waist-up push-ups do not
work at all.

### Measured per-class F1 (held-out subjects, `GroupKFold` by person)

| class | F1 full-body | F1 waist-up | precision (full-body) | recall (full-body) |
|---|---|---|---|---|
| none | 0.902 | 0.855 | — | — |
| wave | 0.918 | 0.873 | 0.906 | 0.930 |
| squat | 0.922 | 0.803 | 0.998 | 0.856 |
| clapping | 0.898 | 0.899 | 0.895 | 0.900 |
| jumping_jacks | 0.887 | 0.861 | 0.986 | 0.805 |
| pushup | 0.381 | 0.137 | 0.941 | 0.239 |

### What to demonstrate, and what not to

- **Demonstrate wave, clapping and jumping jacks.**
- **Squat: yes, but say out loud that it misses about 1 repetition in 7 with the legs in frame, about 1 in 3
  waist-up** — recall 0.856 full-body, 0.685 waist-up. It almost never invents one (precision 0.998); it misses.
- **Do not demonstrate push-ups**, in either framing.

### Push-ups are unmeasured, not proven broken

19 training groups only, with per-fold F1 spanning 0.063 to 0.714 — that spread is the sample size talking,
not a stable score. On HMDB51 the same model reaches F1 **0.707 at precision 1.000**, and misread **0 of
525** situp windows, situps being the confusable neighbour. Push-ups need *data*, not threshold tuning.

### `wave` 0.918 and `clapping` 0.898 must never be quoted as robot performance

Both come from NTU RGB+D, a studio corpus, and a source confound inflates them: `none` recall is **0.770**
on studio NTU against **0.977** on handheld UCF101, i.e. the model over-fires the two studio-only classes on
studio negatives. The magnitude is **bracketed, not measured** — no corpus we have varies the source while
holding framing and video quality fixed — but the bracket is wide: `wave` scores 0.918 in-domain, **0.581**
on our own robot recordings, and 0.176 on HMDB51. **For a robot in a room, quote 0.581.**

One measurement **gap**, stated as a gap: `clapping` has **never been measured on robot footage**. Our own
recordings contain waves and pauses only, so unlike `wave` — which also has a dedicated model measured at
F1 0.951 on those recordings — clapping's confound inflation is entirely unhedged. One short clapping
recording would close this.

### Data provenance

Training data is HRNet COCO-17 skeletons published by OpenMMLab for NTU RGB+D 60 (wave, clapping) and
UCF101 (push-ups, squats, jumping jacks); HMDB51 is an external test set only. URLs, sizes, sha256, class
resolution and caveats: [`datasets/SOURCES.md`](../datasets/SOURCES.md). **NTU RGB+D requires accepting the
ROSE Lab terms**, and **UCF101 is research-use only**. The pickles are 1.8 GB and gitignored.

```bash
tools/download_datasets.sh
reachy_mini_env/bin/python -m training.actions.train --ntu datasets/ntu60_hrnet.pkl \
    --ucf datasets/ucf101_hrnet.pkl --hmdb datasets/hmdb51_2d.pkl \
    --crop-fraction 0 0.25 0.5 1 --ship 0.25
```

`--crop-fraction` is how much of the waist-up augmentation reaches the **training** half (0 = full-body
windows only, 1 = every cropped copy as well); it never touches the evaluation half, so every setting is
scored on the same windows. `--ship 0.25` is the shipped setting. Without it the waist-up F1 of `wave`, `squat` and
`clapping` is 0.000.

