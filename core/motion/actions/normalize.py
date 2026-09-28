"""Keypoints normalized to the torso, plus `trunk_height`, the one quantity that keeps global translation.

`normalize_to_torso` puts the origin on the shoulder midpoint, which deletes a squat seen waist-up (mostly
vertical translation). `trunk_height` measures the shoulder midpoint's frame position in torso lengths
and needs no legs. Camera motion injects fake oscillation into it, so handheld footage (UCF101) is noisier
than studio footage (NTU).
"""

import numpy as np

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

MIN_TORSO_LENGTH = 1e-3
# A confident-keypoint bounding box smaller than this (frame units) has no recoverable body scale.
MIN_BODY_EXTENT = 0.05
# Shoulder width below this fraction of the bounding-box diagonal means an edge-on view: the shoulders
# carry no scale. Low on purpose: it accepts rotations up to ~82 degrees and keeps ~24x margin above the
# measured degenerate case (0.000826).
MIN_SHOULDER_WIDTH_FRACTION_OF_BODY_EXTENT = 0.02
# Square-on torso_length / shoulder_width, measured on NTU RGB+D 60 (median over the widest-shoulder frame
# of each sequence). Makes the shoulder-width fallback commensurate with the hip-based scale.
SHOULDER_WIDTH_TO_TORSO_LENGTH = 1.666
# `trunk_height` is measured from the frame centre so it is invariant to distance from the camera.
FRAME_CENTRE_Y = 0.5


def torso_scale(keypoints, min_score, aspect_ratio):
    """Body scale in aspect-corrected frame units.

    Shoulder-to-hip distance when both hips are confident, else the shoulder width rescaled by
    `SHOULDER_WIDTH_TO_TORSO_LENGTH`.

    Args:
        keypoints: (17, 3) frame-normalized keypoints, or None.
        min_score: Confidence threshold.
        aspect_ratio: Frame width / height.

    Returns:
        The scale, or None without two confident shoulders, a body too small to scale, or shoulders
        collapsed edge-on with no confident hips.
    """
    if keypoints is None:
        return None
    if keypoints[LEFT_SHOULDER, 2] < min_score or keypoints[RIGHT_SHOULDER, 2] < min_score:
        return None
    xy = keypoints[:, :2] * np.array([aspect_ratio, 1.0], np.float32)
    shoulder_mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    if keypoints[LEFT_HIP, 2] >= min_score and keypoints[RIGHT_HIP, 2] >= min_score:
        hip_mid = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2
        scale = float(np.linalg.norm(hip_mid - shoulder_mid))
    else:
        shoulder_width = float(np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER]))
        confident_xy = xy[keypoints[:, 2] >= min_score]
        diagonal = float(np.linalg.norm(confident_xy.max(axis=0) - confident_xy.min(axis=0)))
        if diagonal < MIN_BODY_EXTENT:
            return None
        if shoulder_width < MIN_SHOULDER_WIDTH_FRACTION_OF_BODY_EXTENT * diagonal:
            return None
        scale = shoulder_width * SHOULDER_WIDTH_TO_TORSO_LENGTH
    return max(scale, MIN_TORSO_LENGTH)


def trunk_height(keypoints, min_score, aspect_ratio):
    """Shoulder midpoint height from the frame centre, in torso lengths (image y down: crouching grows it).

    Returns:
        The height, or nan in the frames `normalize_to_torso` rejects.
    """
    height, scale = trunk_height_parts(keypoints, min_score, aspect_ratio)
    return height / scale


def trunk_height_parts(keypoints, min_score, aspect_ratio):
    """`trunk_height` before the division: (centre-referenced frame y, torso scale), or (nan, nan).

    Kept apart so a window can divide every frame by its *median* scale: the shoulder-width fallback
    projects as cos(rotation), and a per-frame denominator turns shoulder turns into fake vertical motion.
    """
    scale = torso_scale(keypoints, min_score, aspect_ratio)
    if scale is None:
        return float("nan"), float("nan")
    shoulder_mid_y = float(keypoints[LEFT_SHOULDER, 1] + keypoints[RIGHT_SHOULDER, 1]) / 2
    return shoulder_mid_y - FRAME_CENTRE_Y, scale


def normalize_to_torso(keypoints, min_score, aspect_ratio):
    """Express keypoints in torso lengths from the shoulder midpoint (image y points down).

    Args:
        keypoints: (17, 3) frame-normalized keypoints, or None.
        min_score: Confidence threshold.
        aspect_ratio: Frame width / height.

    Returns:
        The normalized (17, 3) array, or None when `torso_scale` finds no credible scale.
    """
    scale = torso_scale(keypoints, min_score, aspect_ratio)
    if scale is None:
        return None
    xy = keypoints[:, :2] * np.array([aspect_ratio, 1.0], np.float32)
    shoulder_mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    normalized = keypoints.astype(np.float32).copy()
    normalized[:, :2] = (xy - shoulder_mid) / scale
    return normalized
