"""Sliding time window of recent poses, normalized to the person's shoulders."""

from collections import deque

import numpy as np

from core.pose_backends import LEFT_SHOULDER, RIGHT_SHOULDER

WINDOW_S = 1.5
MIN_SHOULDER_WIDTH = 1e-3


def normalize_to_shoulders(keypoints, min_score, aspect_ratio):
    """Express keypoints in shoulder widths from the shoulder midpoint.

    Args:
        keypoints: (17, 3) frame-normalized keypoints, or None.
        min_score: Confidence both shoulders must reach.
        aspect_ratio: Frame width / height; x is multiplied by it so both axes share a unit.

    Returns:
        The normalized (17, 3) array (image y points down), or None without two confident shoulders.
    """
    if keypoints is None:
        return None
    if keypoints[LEFT_SHOULDER, 2] < min_score or keypoints[RIGHT_SHOULDER, 2] < min_score:
        return None
    xy = keypoints[:, :2] * np.array([aspect_ratio, 1.0], np.float32)
    mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    width = max(float(np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER])), MIN_SHOULDER_WIDTH)
    normalized = keypoints.astype(np.float32).copy()
    normalized[:, :2] = (xy - mid) / width
    return normalized


class KeypointWindow:
    """Keeps the last `duration_s` seconds of normalized poses."""

    def __init__(self, duration_s=WINDOW_S, min_score=0.5, aspect_ratio=1.0,
                 normalizer=normalize_to_shoulders, extra=None):
        """Configure the window.

        Args:
            duration_s: Window length in seconds.
            min_score: Keypoint confidence threshold passed to the normalizer.
            aspect_ratio: Frame width / height.
            normalizer: `f(keypoints, min_score, aspect_ratio) -> (17, 3) or None`.
            extra: Optional per-frame channel `f(keypoints, min_score, aspect_ratio) -> float or tuple`,
                returning nan(s) when not computable. Used for quantities normalization destroys
                (`trunk_height_parts`).
        """
        self.duration_s = duration_s
        self.min_score = min_score
        self.aspect_ratio = aspect_ratio
        self.normalizer = normalizer
        self.extra = extra
        self._frames = deque()

    def add(self, t, keypoints):
        """Append a frame at time `t` and drop frames older than the window."""
        value = (np.nan if self.extra is None
                 else np.asarray(self.extra(keypoints, self.min_score, self.aspect_ratio), np.float64))
        self._frames.append((t, self.normalizer(keypoints, self.min_score, self.aspect_ratio), value))
        while self._frames[0][0] < t - self.duration_s:
            self._frames.popleft()

    def frames(self):
        """(times, keypoints) arrays; rejected frames are NaN rows in `keypoints`."""
        times = np.array([t for t, _, _extra in self._frames], np.float64)
        keypoints = np.full((len(self._frames), 17, 3), np.nan, np.float32)
        for i, (_, kps, _extra) in enumerate(self._frames):
            if kps is not None:
                keypoints[i] = kps
        return times, keypoints

    def extras(self):
        """The `extra` channel aligned with `frames()`: (frames,) or (frames, k)."""
        return np.array([value for _, _, value in self._frames], np.float64)

    def span(self):
        """Seconds between the oldest and newest frame."""
        return self._frames[-1][0] - self._frames[0][0] if len(self._frames) >= 2 else 0.0

    def valid_frac(self):
        """Fraction of frames the normalizer accepted."""
        if not self._frames:
            return 0.0
        return sum(kps is not None for _, kps, _extra in self._frames) / len(self._frames)
