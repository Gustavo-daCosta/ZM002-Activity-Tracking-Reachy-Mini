"""Sliding window of recent poses, normalized to the person's body.

Two normalizers exist. `normalize_to_shoulders` (wave pipeline) uses the shoulder width as the unit;
`normalize_to_torso` (action pipeline) uses the shoulder-to-hip length, which survives a waist-up framing.
Both put the origin on the shoulder midpoint, which deletes vertical translation; `trunk_height_parts` is
the one channel that keeps it, so a squat seen waist-up is still measurable.
"""

from collections import deque

import numpy as np

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

WINDOW_S = 1.5
ACTION_WINDOW_S = 3.0  # a push-up or squat repetition takes 2-3 s
MIN_SHOULDER_WIDTH = 1e-3
MIN_TORSO_LENGTH = 1e-3
# A confident-keypoint bounding box smaller than this (frame units) has no recoverable body scale.
MIN_BODY_EXTENT = 0.05
# Shoulder width below this fraction of the body diagonal means an edge-on view with no scale in it.
MIN_SHOULDER_WIDTH_FRACTION = 0.02
# Square-on torso_length / shoulder_width, median over NTU RGB+D 60.
SHOULDER_WIDTH_TO_TORSO_LENGTH = 1.666
FRAME_CENTRE_Y = 0.5


def _aspect_corrected(keypoints, aspect_ratio):
    return keypoints[:, :2] * np.array([aspect_ratio, 1.0], np.float32)


def normalize_to_shoulders(keypoints, min_score, aspect_ratio):
    """Express keypoints in shoulder widths from the shoulder midpoint.

    Args:
        keypoints: (17, 3) frame-normalized keypoints, or None.
        min_score: Confidence both shoulders must reach.
        aspect_ratio: Frame width / height; x is multiplied by it so both axes share a unit.

    Returns:
        The normalized (17, 3) array (image y points down), or None without two confident shoulders.
    """
    if keypoints is None or keypoints[LEFT_SHOULDER, 2] < min_score or keypoints[RIGHT_SHOULDER, 2] < min_score:
        return None
    xy = _aspect_corrected(keypoints, aspect_ratio)
    mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    width = max(float(np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER])), MIN_SHOULDER_WIDTH)
    normalized = keypoints.astype(np.float32).copy()
    normalized[:, :2] = (xy - mid) / width
    return normalized


def torso_scale(keypoints, min_score, aspect_ratio):
    """Body scale in aspect-corrected frame units: shoulder-to-hip length, else rescaled shoulder width.

    Returns:
        The scale, or None without two confident shoulders, a body too small to scale, or shoulders
        collapsed edge-on with no confident hips.
    """
    if keypoints is None or keypoints[LEFT_SHOULDER, 2] < min_score or keypoints[RIGHT_SHOULDER, 2] < min_score:
        return None
    xy = _aspect_corrected(keypoints, aspect_ratio)
    shoulder_mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    if keypoints[LEFT_HIP, 2] >= min_score and keypoints[RIGHT_HIP, 2] >= min_score:
        scale = float(np.linalg.norm((xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2 - shoulder_mid))
    else:
        shoulder_width = float(np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER]))
        confident_xy = xy[keypoints[:, 2] >= min_score]
        diagonal = float(np.linalg.norm(confident_xy.max(axis=0) - confident_xy.min(axis=0)))
        if diagonal < MIN_BODY_EXTENT or shoulder_width < MIN_SHOULDER_WIDTH_FRACTION * diagonal:
            return None
        scale = shoulder_width * SHOULDER_WIDTH_TO_TORSO_LENGTH
    return max(scale, MIN_TORSO_LENGTH)


def normalize_to_torso(keypoints, min_score, aspect_ratio):
    """Express keypoints in torso lengths from the shoulder midpoint, or None without a credible scale."""
    scale = torso_scale(keypoints, min_score, aspect_ratio)
    if scale is None:
        return None
    xy = _aspect_corrected(keypoints, aspect_ratio)
    normalized = keypoints.astype(np.float32).copy()
    normalized[:, :2] = (xy - (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2) / scale
    return normalized


def trunk_height_parts(keypoints, min_score, aspect_ratio):
    """(shoulder-midpoint y from the frame centre, torso scale), or (nan, nan) when rejected.

    Kept apart so a window divides every frame by its median scale: a per-frame denominator turns
    shoulder turns into fake vertical motion.
    """
    scale = torso_scale(keypoints, min_score, aspect_ratio)
    if scale is None:
        return float("nan"), float("nan")
    return float(keypoints[LEFT_SHOULDER, 1] + keypoints[RIGHT_SHOULDER, 1]) / 2 - FRAME_CENTRE_Y, scale


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
            extra: Optional per-frame channel with the same signature returning a float or tuple
                (nan when not computable), e.g. `trunk_height_parts`.
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
        times = np.array([t for t, _, _ in self._frames], np.float64)
        keypoints = np.full((len(self._frames), 17, 3), np.nan, np.float32)
        for i, (_, kps, _) in enumerate(self._frames):
            if kps is not None:
                keypoints[i] = kps
        return times, keypoints

    def extras(self):
        """The `extra` channel aligned with `frames()`: (frames,) or (frames, k)."""
        return np.array([value for _, _, value in self._frames], np.float64)

    def span(self):
        """Seconds between the oldest and newest frame."""
        return self._frames[-1][0] - self._frames[0][0] if len(self._frames) >= 2 else 0.0
