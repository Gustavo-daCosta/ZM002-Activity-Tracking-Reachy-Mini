"""Recorded sessions (.npz keypoint sequences with round labels) and training windows."""

from pathlib import Path

import numpy as np

from core.motion.features import window_features
from core.motion.window import WINDOW_S, KeypointWindow, normalize_to_shoulders

DATA_DIR = Path(__file__).resolve().parent / "data" / "wave"
STEP_S = 0.25


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
    np.savez_compressed(
        path,
        t=np.asarray(t, np.float64),
        keypoints=np.asarray(keypoints, np.float32),
        labels=np.asarray(labels, np.int8),
        rounds=np.asarray(rounds, np.int16),
        pose_model=np.array(pose_model),
        aspect_ratio=np.float64(aspect_ratio),
    )


def load_session(path):
    """Read a session written by `save_session` into a dict."""
    with np.load(path) as data:
        return {
            "t": data["t"],
            "keypoints": data["keypoints"],
            "labels": data["labels"],
            "rounds": data["rounds"],
            "pose_model": str(data["pose_model"]),
            "aspect_ratio": float(data["aspect_ratio"]),
        }


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

    Yields:
        Feature dicts (None results are skipped).
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


def make_windows(session, window_s=WINDOW_S, step_s=STEP_S, min_score=0.5,
                 features_of=window_features, normalizer=normalize_to_shoulders, extra=None):
    """Feature windows every `step_s`, each built only from frames of a single labeled round.

    The defaults reproduce the wave pipeline; the action pipeline passes `action_features`,
    `normalize_to_torso` and `trunk_height_parts`.

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
