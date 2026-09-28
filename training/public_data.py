"""Wave windows from public skeleton datasets, in the pkl format published by MMAction2 / PySkl.

    NTU RGB+D 60   ntu60_hrnet.pkl   56 880 clips, 60 classes, class 22 = "A23 hand waving", 40 subjects
    HMDB51         hmdb51_2d.pkl      6 371 clips, 51 classes, class 50 = "wave" (movies, in the wild)

    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
    https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl

Every clip is resampled from 30 FPS to the robot's ~10 FPS before windowing, because features like the
reversal count depend on how densely the motion is sampled.
"""

import pickle
import re
from pathlib import Path

import numpy as np

from core.motion.features import window_features
from core.motion.window import WINDOW_S, KeypointWindow, normalize_to_shoulders
from training.dataset import emit_windows

NTU_WAVE_LABEL = 22
HMDB_WAVE_LABEL = 50
SOURCE_FPS = 30.0
ROBOT_FPS = 10.0
STEP_S = 0.5
MIN_SCORE = 0.5

NTU_SUBJECT = re.compile(r"S\d+C\d+(P\d+)R\d+A\d+")


def ntu_subject(frame_dir: str) -> str:
    """The subject id encoded in an NTU clip name; anything else is its own group."""
    match = NTU_SUBJECT.match(frame_dir)
    return match.group(1) if match else frame_dir


def _person_index(annotation) -> int:
    """The person with the most confident keypoints (HMDB51 clips can hold a crowd), or -1."""
    scores = annotation["keypoint_score"]
    return int(np.argmax(scores.mean(axis=(1, 2)))) if len(scores) else -1


def clip_windows(annotation, source_fps=SOURCE_FPS, target_fps=ROBOT_FPS, window_s=WINDOW_S,
                 step_s=STEP_S, min_score=MIN_SCORE, features_of=window_features,
                 normalizer=normalize_to_shoulders, extra=None, min_span_s=None):
    """Feature windows of one clip, keypoints normalized to the frame like the robot pipeline does.

    Args:
        annotation: One PySkl annotation dict.
        source_fps: Dataset frame rate.
        target_fps: Frame rate to resample to.
        window_s: Window length.
        step_s: Seconds between windows.
        min_score: Confidence threshold.
        features_of: Feature function (`action_features` for the action pipeline).
        normalizer: Keypoint normalizer.
        extra: `KeypointWindow` extra channel.
        min_span_s: Span before the first window; defaults to `window_s - step_s`. The action loaders
            lower it to `MIN_SPAN_S`, since the default discarded 47% of NTU's short clips.

    Returns:
        List of feature dicts.
    """
    person = _person_index(annotation)
    if person < 0:
        return []
    height, width = annotation["img_shape"]
    keypoints = annotation["keypoint"][person]
    scores = annotation["keypoint_score"][person]
    stride = max(1, round(source_fps / target_fps))
    min_span_s = window_s - step_s if min_span_s is None else min_span_s

    def frames():
        for index in range(0, len(keypoints), stride):
            frame = np.zeros((17, 3), np.float32)
            # float16 in the published files: cast before dividing.
            frame[:, 0] = np.asarray(keypoints[index, :, 0], np.float64) / width
            frame[:, 1] = np.asarray(keypoints[index, :, 1], np.float64) / height
            frame[:, 2] = scores[index]
            yield index / source_fps, frame

    window = KeypointWindow(window_s, min_score, width / height, normalizer=normalizer, extra=extra)
    return list(emit_windows(window, frames(), step_s, min_span_s, features_of, min_score))


def load_annotations(path):
    """The `annotations` list of a PySkl pickle."""
    with open(Path(path), "rb") as handle:
        return pickle.load(handle)["annotations"]


def dataset_windows(path, wave_label, max_clips_per_class=None, max_wave_clips=None,
                    group_of=ntu_subject, **kwargs):
    """(features, label, group) for every clip: label 1 for the wave class, 0 for every other class.

    Args:
        path: The pickle.
        wave_label: Integer label of the wave class.
        max_clips_per_class: Cap per negative class, keeping the negatives balanced.
        max_wave_clips: Cap for the wave class (default: all).
        group_of: `frame_dir -> group key` for grouped validation.
        **kwargs: Passed to `clip_windows`.
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
