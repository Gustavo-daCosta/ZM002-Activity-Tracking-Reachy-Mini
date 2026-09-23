"""Recorded sessions (.npz keypoint sequences with round labels) and training windows."""

from pathlib import Path

import numpy as np

from core.motion import OTHER, PAUSE, WAVE  # noqa: F401  (re-exported for record/train)
from core.motion.features import window_features
from core.motion.window import WINDOW_S, KeypointWindow, normalize_to_shoulders

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "wave"
STEP_S = 0.25


def save_session(path, t, keypoints, labels, rounds, pose_model, aspect_ratio):
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
    return sorted(Path(data_dir).glob("session_*.npz"))


def make_windows(session, window_s=WINDOW_S, step_s=STEP_S, min_score=0.5,
                 features_of=window_features, normalizer=normalize_to_shoulders, extra=None):
    """Feature windows every step_s, each built only from frames of a single labeled round.

    The defaults reproduce the wave pipeline exactly. `features_of`, `normalizer` and `extra` are the same
    three parameters `public_data.clip_windows` takes, for the same reason and with the same meaning: the
    action pipeline needs a 3 s window of torso-normalized keypoints described by
    `actions.features.action_features`, with `actions.normalize.trunk_height_parts` collected as an extra
    per-frame channel and passed on as `trunk_heights=`.

    `extra` in particular is not optional polish. Two of the action features -- `trunk_height_amplitude`
    and `trunk_height_reversal_rate` -- are computed only from that channel, so evaluating a model on these
    recordings without it would silently replace both with their 0.0 "not measured" sentinel and compare a
    34-column training vector against a de facto 32-column one. With `extra=None` no extra keyword is
    passed at all, so `window_features`, which does not accept one, still works.
    """
    t, keypoints, labels, rounds = session["t"], session["keypoints"], session["labels"], session["rounds"]
    windows = []
    for round_id in np.unique(rounds[rounds >= 0]):
        indices = np.flatnonzero(rounds == round_id)
        label = int(labels[indices[0]])
        window = KeypointWindow(window_s, min_score, session["aspect_ratio"],
                                normalizer=normalizer, extra=extra)
        next_emit = None
        for i in indices:
            window.add(float(t[i]), keypoints[i])
            if window.span() < window_s - step_s:
                continue
            if next_emit is None or t[i] >= next_emit:
                kwargs = {} if extra is None else {"trunk_heights": window.extras()}
                features = features_of(*window.frames(), min_score=min_score, **kwargs)
                if features is not None:
                    windows.append((features, label, int(round_id)))
                next_emit = t[i] + step_s
    return windows
