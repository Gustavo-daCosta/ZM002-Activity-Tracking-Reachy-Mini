"""Training windows: from our recorded sessions and from the public skeleton datasets.

Recorded sessions are `.npz` keypoint sequences with per-frame round labels (training/record.py). Public
data is the COCO-17 skeleton pickles published by MMAction2 / PySkl (datasets/SOURCES.md), resampled from
30 FPS to the robot's ~10 FPS before windowing because features like the reversal count depend on how
densely the motion is sampled. Both go through the same `KeypointWindow` the live monitors use.
"""

import pickle
import re
from pathlib import Path

import numpy as np

from core.motion.features import ACTION_MIN_SPAN_S, action_features, window_features
from core.motion.forest import NONE
from core.motion.window import (
    ACTION_WINDOW_S, WINDOW_S, KeypointWindow, normalize_to_shoulders, normalize_to_torso, trunk_height_parts,
)
from core.pose_backends import LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE

DATA_DIR = Path(__file__).resolve().parent / "data" / "wave"
STEP_S = 0.25
PUBLIC_STEP_S = 0.5
SOURCE_FPS = 30.0
ROBOT_FPS = 10.0
MIN_SCORE = 0.5

NTU_WAVE_LABEL = 22  # "A23 hand waving"
HMDB_WAVE_LABEL = 50
NTU_LABELS = {22: "wave", 9: "clapping"}  # A23 hand waving, A10 clapping (0-based)
# HMDB51 classes are alphabetical. 38 = situp is deliberately not mapped so a situp called a pushup stays
# the false positive it is. External test set only.
HMDB_LABELS = {50: "wave", 29: "pushup", 4: "clapping"}
UCF_LABELS = {"PushUps": "pushup", "BodyWeightSquats": "squat", "JumpingJack": "jumping_jacks"}
NTU_SUBJECT = re.compile(r"S\d+C\d+(P\d+)R\d+A\d+")
UCF_NAME = re.compile(r"v_([A-Za-z]+)_(g\d+)_c\d+")

# Crop augmentation: each clip is windowed as published and once per cropped regime (the robot sees a
# waist-up person; the datasets never do). Two levels because "legs" and "waist" take different scale
# paths in `normalize_to_torso`.
CROP_LEVELS = {
    "full": (),
    "legs": (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE),
    "waist": (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE),
}

ACTION_WINDOW = dict(window_s=ACTION_WINDOW_S, features_of=action_features, normalizer=normalize_to_torso,
                     extra=trunk_height_parts)


# --- recorded sessions -------------------------------------------------------------------------------

def save_session(path, t, keypoints, labels, rounds, pose_model, aspect_ratio):
    """Write a recorded session as a compressed `.npz`.

    Args:
        path: Output file.
        t: Timestamps in seconds.
        keypoints: (frames, 17, 3) keypoints.
        labels: Per-frame label (WAVE / OTHER / PAUSE).
        rounds: Per-frame round id (-1 during pauses).
        pose_model: Backend name used to record.
        aspect_ratio: Frame width / height.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=np.asarray(t, np.float64), keypoints=np.asarray(keypoints, np.float32),
                        labels=np.asarray(labels, np.int8), rounds=np.asarray(rounds, np.int16),
                        pose_model=np.array(pose_model), aspect_ratio=np.float64(aspect_ratio))


def load_session(path):
    """Read a session written by `save_session` into a dict."""
    with np.load(path) as data:
        return {"t": data["t"], "keypoints": data["keypoints"], "labels": data["labels"], "rounds": data["rounds"],
                "pose_model": str(data["pose_model"]), "aspect_ratio": float(data["aspect_ratio"])}


def list_sessions(data_dir=DATA_DIR):
    """Session files in `data_dir`, sorted."""
    return sorted(Path(data_dir).glob("session_*.npz"))


def emit_windows(window, frames, step_s, min_span_s, features_of, min_score):
    """Feed frames into a `KeypointWindow` and yield a feature dict every `step_s` seconds.

    Args:
        window: The window (its `extra` channel, when set, is passed to `features_of` as `trunk_heights`).
        frames: Iterable of (timestamp, keypoints).
        step_s: Seconds between emitted windows.
        min_span_s: Span the window must reach before the first emission.
        features_of: `f(times, keypoints, min_score=..., [trunk_heights=...]) -> dict or None`.
        min_score: Confidence threshold passed to `features_of`.
    """
    next_emit = None
    for timestamp, keypoints in frames:
        window.add(timestamp, keypoints)
        if window.span() < min_span_s:
            continue
        if next_emit is None or timestamp >= next_emit:
            kwargs = {} if window.extra is None else {"trunk_heights": window.extras()}
            features = features_of(*window.frames(), min_score=min_score, **kwargs)
            if features is not None:
                yield features
            next_emit = timestamp + step_s


def make_windows(session, window_s=WINDOW_S, step_s=STEP_S, min_score=MIN_SCORE,
                 features_of=window_features, normalizer=normalize_to_shoulders, extra=None):
    """Feature windows every `step_s`, each built only from frames of a single labeled round.

    The defaults reproduce the wave pipeline; `**ACTION_WINDOW` selects the action pipeline.

    Returns:
        List of (features, label, round_id).
    """
    t, keypoints, labels, rounds = session["t"], session["keypoints"], session["labels"], session["rounds"]
    windows = []
    for round_id in np.unique(rounds[rounds >= 0]):
        indices = np.flatnonzero(rounds == round_id)
        label = int(labels[indices[0]])
        window = KeypointWindow(window_s, min_score, session["aspect_ratio"], normalizer=normalizer, extra=extra)
        frames = ((float(t[i]), keypoints[i]) for i in indices)
        for features in emit_windows(window, frames, step_s, window_s - step_s, features_of, min_score):
            windows.append((features, label, int(round_id)))
    return windows


# --- public skeleton datasets ------------------------------------------------------------------------

def load_annotations(path):
    """The `annotations` list of a PySkl pickle."""
    with open(Path(path), "rb") as handle:
        return pickle.load(handle)["annotations"]


def ntu_subject(frame_dir):
    """The subject id encoded in an NTU clip name; anything else is its own group."""
    match = NTU_SUBJECT.match(frame_dir)
    return match.group(1) if match else frame_dir


def ucf_class(frame_dir):
    """`PushUps` from `v_PushUps_g08_c02`; the whole name when it does not match."""
    match = UCF_NAME.match(frame_dir)
    return match.group(1) if match else frame_dir


def ucf_group(frame_dir):
    """Group key of a UCF101 clip: clips cut from the same footage share `g<n>`."""
    match = UCF_NAME.match(frame_dir)
    return f"ucf:{match.group(1)}_{match.group(2)}" if match else f"ucf:{frame_dir}"


def clip_windows(annotation, window_s=WINDOW_S, step_s=PUBLIC_STEP_S, min_score=MIN_SCORE,
                 features_of=window_features, normalizer=normalize_to_shoulders, extra=None, min_span_s=None):
    """Feature windows of one clip, keypoints normalized to the frame like the robot pipeline does.

    The person with the most confident keypoints is used (HMDB51 clips can hold a crowd). `min_span_s`
    defaults to `window_s - step_s`; the action loaders lower it, since the default discarded 47% of
    NTU's short clips.
    """
    scores_all = annotation["keypoint_score"]
    if not len(scores_all):
        return []
    person = int(np.argmax(scores_all.mean(axis=(1, 2))))
    height, width = annotation["img_shape"]
    keypoints, scores = annotation["keypoint"][person], scores_all[person]
    stride = max(1, round(SOURCE_FPS / ROBOT_FPS))
    min_span_s = window_s - step_s if min_span_s is None else min_span_s

    def frames():
        for index in range(0, len(keypoints), stride):
            frame = np.zeros((17, 3), np.float32)
            frame[:, 0] = np.asarray(keypoints[index, :, 0], np.float64) / width  # float16 in the files
            frame[:, 1] = np.asarray(keypoints[index, :, 1], np.float64) / height
            frame[:, 2] = scores[index]
            yield index / SOURCE_FPS, frame

    window = KeypointWindow(window_s, min_score, width / height, normalizer=normalizer, extra=extra)
    return list(emit_windows(window, frames(), step_s, min_span_s, features_of, min_score))


def wave_dataset_windows(path, wave_label, max_clips_per_class=None):
    """(features, label, subject) for every clip: label 1 for the wave class, 0 for every other class.

    `max_clips_per_class` caps the negative classes so they stay balanced; the wave class is never capped.
    """
    taken, windows = {}, []
    for annotation in load_annotations(path):
        label = annotation["label"]
        if label != wave_label and max_clips_per_class is not None and taken.get(label, 0) >= max_clips_per_class:
            continue
        clip = clip_windows(annotation)
        if not clip:
            continue
        taken[label] = taken.get(label, 0) + 1
        group = ntu_subject(annotation["frame_dir"])
        windows += [(features, int(label == wave_label), group) for features in clip]
    return windows


def _crop_scores(annotation, joints):
    """A shallow copy of `annotation` with `joints` made unconfident, so the pipeline sees a cropped skeleton."""
    if not joints:
        return annotation
    scores = annotation["keypoint_score"].copy()
    scores[:, :, list(joints)] = 0.0
    return dict(annotation, keypoint_score=scores)


def _action_windows(path, label_of, group_of, class_of, max_clips_per_class):
    """(features, action name, group) for every clip at every crop level, capping clips per negative class.

    Positives are never capped: there are far fewer of them than negatives.
    """
    taken, out = {}, []
    for annotation in load_annotations(path):
        action = label_of(annotation)
        if action == NONE and max_clips_per_class is not None and taken.get(class_of(annotation), 0) >= max_clips_per_class:
            continue
        clip = [dict(features, crop=level)
                for level, joints in CROP_LEVELS.items()
                for features in clip_windows(_crop_scores(annotation, joints), min_span_s=ACTION_MIN_SPAN_S, **ACTION_WINDOW)]
        if not clip:
            continue
        taken[class_of(annotation)] = taken.get(class_of(annotation), 0) + 1
        group = group_of(annotation["frame_dir"])
        out += [(features, action, group) for features in clip]
    return out


def ntu_windows(path, max_clips_per_class=None):
    """Action windows from ntu60_hrnet.pkl: waving and clapping, everything else `none`; grouped by subject."""
    return _action_windows(path, lambda a: NTU_LABELS.get(a["label"], NONE), lambda d: f"ntu:{ntu_subject(d)}",
                           lambda a: ("ntu", a["label"]), max_clips_per_class)


def hmdb_windows(path, max_clips_per_class=None):
    """Action windows from hmdb51_2d.pkl (movies; external test set only)."""
    return _action_windows(path, lambda a: HMDB_LABELS.get(a["label"], NONE), lambda d: f"hmdb:{d}",
                           lambda a: ("hmdb", a["label"]), max_clips_per_class)


def ucf_windows(path, max_clips_per_class=None):
    """Action windows from ucf101_hrnet.pkl: push-ups, squats and jumping jacks, everything else `none`."""
    return _action_windows(path, lambda a: UCF_LABELS.get(ucf_class(a["frame_dir"]), NONE), ucf_group,
                           lambda a: ("ucf", ucf_class(a["frame_dir"])), max_clips_per_class)


def recorded_action_windows(data_dir=DATA_DIR):
    """Our own recordings as action windows (waves and pauses only), an external test set."""
    out = []
    for index, path in enumerate(list_sessions(data_dir)):
        for features, label, _ in make_windows(load_session(path), **ACTION_WINDOW):
            out.append((dict(features, crop="recorded"), "wave" if label == 1 else NONE, f"rec:session{index}"))
    return out
