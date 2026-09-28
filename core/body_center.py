"""Pick the point the robot should follow: torso center, else shoulders, else nose."""

from typing import Optional, Tuple

import numpy as np

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER


def body_center(keypoints: Optional[np.ndarray], min_score: float) -> Optional[Tuple[float, float, str]]:
    """Choose the tracking target from the confident keypoints.

    Args:
        keypoints: (17, 3) normalized keypoints, or None.
        min_score: Confidence threshold.

    Returns:
        (x, y, source) with source "torso", "shoulders" or "nose", or None if nothing is confident.
    """
    if keypoints is None:
        return None

    def confident(*indices):
        return [i for i in indices if keypoints[i, 2] >= min_score]

    shoulders = confident(LEFT_SHOULDER, RIGHT_SHOULDER)
    hips = confident(LEFT_HIP, RIGHT_HIP)
    if shoulders and hips:
        indices, source = shoulders + hips, "torso"
    elif shoulders:
        indices, source = shoulders, "shoulders"
    elif confident(NOSE):
        indices, source = [NOSE], "nose"
    else:
        return None
    x, y = keypoints[indices, :2].mean(axis=0)
    return float(x), float(y), source
