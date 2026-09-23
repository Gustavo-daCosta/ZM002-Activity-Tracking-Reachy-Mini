"""Sliding time window of recent poses, normalized to the person's shoulders."""

from collections import deque

import numpy as np

from core.pose_backends import LEFT_SHOULDER, RIGHT_SHOULDER

WINDOW_S = 1.5
MIN_SHOULDER_WIDTH = 1e-3


def normalize_to_shoulders(keypoints, min_score, aspect_ratio):
    """Frame-normalized (17, 3) keypoints -> x, y in shoulder widths from the shoulder midpoint.

    x is first multiplied by the frame aspect ratio (width / height) so both axes use the same unit.
    Image y points down: negative y is above the shoulders. Returns None without two confident shoulders.
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
    def __init__(self, duration_s=WINDOW_S, min_score=0.5, aspect_ratio=1.0,
                 normalizer=normalize_to_shoulders, extra=None):
        """`extra(keypoints, min_score, aspect_ratio)` is an optional extra per-frame channel.

        It must return a float, or a fixed-length tuple of floats, for **every** frame; `nan` (or a tuple
        of `nan`) is how it reports "not computable here". Returning `None` is illegal and raises
        `TypeError` in `np.asarray(..., float)` rather than being silently absorbed -- a channel that
        sometimes yields nothing would misalign with `frames()`, and nan is the aligned way to say it.

        It exists for quantities that normalization deliberately destroys -- `trunk_height_parts`, which
        needs the shoulder midpoint's position in the *frame* and so cannot be recovered from the
        normalized keypoints at all. With `extra=None` (every existing caller, including the whole wave
        path) nothing is computed and `extras()` is an all-nan array of one value per frame, so no existing
        behavior moves.
        """
        self.duration_s = duration_s
        self.min_score = min_score
        self.aspect_ratio = aspect_ratio
        self.normalizer = normalizer
        self.extra = extra
        self._frames = deque()

    def add(self, t, keypoints):
        value = (np.nan if self.extra is None
                 else np.asarray(self.extra(keypoints, self.min_score, self.aspect_ratio), np.float64))
        self._frames.append((t, self.normalizer(keypoints, self.min_score, self.aspect_ratio), value))
        while self._frames[0][0] < t - self.duration_s:
            self._frames.popleft()

    def frames(self):
        times = np.array([t for t, _, _extra in self._frames], np.float64)
        keypoints = np.full((len(self._frames), 17, 3), np.nan, np.float32)
        for i, (_, kps, _extra) in enumerate(self._frames):
            if kps is not None:
                keypoints[i] = kps
        return times, keypoints

    def extras(self):
        """The `extra` channel aligned with `frames()`: nan where it was not computable, or all-nan.

        Shape `(frames,)` for a scalar channel and for `extra=None`, and `(frames, k)` for a channel that
        returns k-tuples -- the first axis is always one entry per frame, in `frames()` order.
        """
        return np.array([value for _, _, value in self._frames], np.float64)

    def span(self):
        return self._frames[-1][0] - self._frames[0][0] if len(self._frames) >= 2 else 0.0

    def valid_frac(self):
        if not self._frames:
            return 0.0
        return sum(kps is not None for _, kps, _extra in self._frames) / len(self._frames)

    def clear(self):
        self._frames.clear()
