"""Wave windows from public skeleton datasets, in the pkl format published by MMAction2 / PySkl.

Our own recordings are two sessions of one person in front of one camera. These files give the same
COCO-17 2D keypoints for thousands of clips and dozens of subjects, with no video to download and no pose
estimation to run:

    NTU RGB+D 60   ntu60_hrnet.pkl   56 880 clips, 60 classes, class 22 = "A23 hand waving", 40 subjects
    HMDB51         hmdb51_2d.pkl      6 371 clips, 51 classes, class 50 = "wave" (movies, in the wild)

    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
    https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl

Every clip is resampled to the robot's frame rate before windowing: the robot sees ~9-10 FPS and the
datasets are 30 FPS, and features like the wrist reversal count depend on how densely the motion is sampled.
"""

import pickle
import re
from pathlib import Path

import numpy as np

from core.motion.features import window_features
from core.motion.window import WINDOW_S, KeypointWindow, normalize_to_shoulders

NTU_WAVE_LABEL = 22    # A23 hand waving (0-based)
HMDB_WAVE_LABEL = 50   # "wave" (the classes are alphabetical)
SOURCE_FPS = 30.0      # both datasets
ROBOT_FPS = 10.0       # what robot/apps/wave_antennas.py measures on the robot
STEP_S = 0.5
MIN_SCORE = 0.5

NTU_SUBJECT = re.compile(r"S\d+C\d+(P\d+)R\d+A\d+")


def ntu_subject(frame_dir: str) -> str:
    """NTU clip names encode setup/camera/subject/repetition/action; anything else is its own group."""
    match = NTU_SUBJECT.match(frame_dir)
    return match.group(1) if match else frame_dir


def _person_index(annotation) -> int:
    """The person with the most confident keypoints (HMDB51 clips can hold a whole crowd)."""
    scores = annotation["keypoint_score"]
    return int(np.argmax(scores.mean(axis=(1, 2)))) if len(scores) else -1


def clip_windows(annotation, source_fps=SOURCE_FPS, target_fps=ROBOT_FPS, window_s=WINDOW_S,
                 step_s=STEP_S, min_score=MIN_SCORE, features_of=window_features,
                 normalizer=normalize_to_shoulders, extra=None, min_span_s=None):
    """Feature windows of one clip, keypoints normalized to the frame like the robot pipeline does.

    The defaults reproduce the wave pipeline exactly. `features_of`, `window_s`, `normalizer` and `extra`
    are what the action pipeline overrides: a 3 s window of torso-normalized keypoints described by
    `actions.features.action_features`. `extra` is a `KeypointWindow` extra channel
    (`actions.normalize.trunk_height_parts`); when it is given, the collected channel is passed to
    `features_of` as `trunk_heights=`, because that quantity is destroyed by normalization and cannot be
    recovered from the normalized keypoints. With `extra=None` no extra keyword is passed at all, so a
    feature function that does not accept one still works.

    `min_span_s` is how much of the window must have accumulated before the first window is emitted; it
    defaults to `window_s - step_s`, which is the wave pipeline's behavior, unchanged. The action loaders
    lower it to `actions.features.MIN_SPAN_S`, because that is what `action_features` itself already
    accepts and the stricter default was discarding whole clips shorter than 2.5 s -- 47% of NTU, which
    cost more than half of the subjects the dataset exists to provide. Emitted windows then span a range
    rather than one value, which matters for the raw `*_reversals` counts (`*_reversal_rate` is
    normalized); the live robot also produces partial windows while its buffer fills and whenever
    `normalize_to_torso` rejects frames, so a span distribution is if anything closer to deployment.
    """
    person = _person_index(annotation)
    if person < 0:
        return []
    height, width = annotation["img_shape"]
    keypoints = annotation["keypoint"][person]
    scores = annotation["keypoint_score"][person]
    stride = max(1, round(source_fps / target_fps))
    min_span_s = window_s - step_s if min_span_s is None else min_span_s

    window = KeypointWindow(window_s, min_score, width / height, normalizer=normalizer, extra=extra)
    windows, next_emit = [], None
    for index in range(0, len(keypoints), stride):
        frame = np.zeros((17, 3), np.float32)
        # float16 in the published files: cast before dividing so nothing overflows or loses precision.
        frame[:, 0] = np.asarray(keypoints[index, :, 0], np.float64) / width
        frame[:, 1] = np.asarray(keypoints[index, :, 1], np.float64) / height
        frame[:, 2] = scores[index]
        timestamp = index / source_fps
        window.add(timestamp, frame)
        if window.span() < min_span_s:
            continue
        if next_emit is None or timestamp >= next_emit:
            kwargs = {} if extra is None else {"trunk_heights": window.extras()}
            features = features_of(*window.frames(), min_score=min_score, **kwargs)
            if features is not None:
                windows.append(features)
            next_emit = timestamp + step_s
    return windows


def load_annotations(path):
    with open(Path(path), "rb") as handle:
        return pickle.load(handle)["annotations"]


def dataset_windows(path, wave_label, max_clips_per_class=None, max_wave_clips=None,
                    group_of=ntu_subject, **kwargs):
    """(features, label, group) for every clip: label 1 for the wave class, 0 for every other class.

    `max_clips_per_class` caps how many clips each action class contributes, which keeps the negatives
    balanced across the other actions instead of being dominated by the longest ones. The wave class has
    its own cap (`max_wave_clips`, default: keep them all) - there are far fewer positives than negatives.
    """
    annotations = load_annotations(path)
    taken, windows = {}, []
    for annotation in annotations:
        label = annotation["label"]
        cap = max_wave_clips if label == wave_label else max_clips_per_class
        if cap is not None and taken.get(label, 0) >= cap:
            continue
        clip = clip_windows(annotation, **kwargs)
        if not clip:
            continue
        taken[label] = taken.get(label, 0) + 1
        group = group_of(annotation["frame_dir"])
        windows += [(features, int(label == wave_label), group) for features in clip]
    return windows
