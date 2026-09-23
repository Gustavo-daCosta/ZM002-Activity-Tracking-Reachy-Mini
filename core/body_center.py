"""Pick the point the robot should follow: torso center, else shoulders, else nose."""

from typing import Optional, Tuple

import numpy as np

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER


def body_center(keypoints: Optional[np.ndarray], min_score: float) -> Optional[Tuple[float, float, str]]:
    """Return (x, y, source) in normalized image coordinates, or None if nothing usable is confident.

    torso: at least one confident shoulder and one confident hip -> mean of the confident ones.
    shoulders: hips missing (common when sitting close to a laptop) -> mean of confident shoulders.
    nose: last resort.
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
