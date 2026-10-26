"""CMAS items 7 and 8 as a rule on single frames: arms straight above the head, and how long they stay up.

Not a classifier: a held posture has no motion for a window to learn, and the CMAS criteria are geometric
already. Items 12 and 14 (chair rise, pick-up) are the classifier of `training/train_cmas.py`.
COCO-17 has no top-of-head point, so it is extrapolated from the nose and the shoulder line (`HEAD_TOP`).
"""

import numpy as np

from core.pose_backends import (
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
)

# Nose-to-crown height as a fraction of nose-to-shoulder-line height. ponytail: one adult ratio, picked by
# eye; check it against the arms_overhead rounds of collect/recordings_cmas (and on children) before
# trusting the timer near the threshold.
HEAD_TOP = 0.5
STRAIGHT_ELBOW_DEG = 160.0  # "elbows in full extension", with room for keypoint jitter
GRACE_S = 0.5  # one bad frame must not restart the hold clock
GREEN, GREY = (0, 255, 0), (200, 200, 200)
ARM_JOINTS = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)


def _elbow_angle(xy, shoulder, elbow, wrist):
    a, b = xy[shoulder] - xy[elbow], xy[wrist] - xy[elbow]
    cosine = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def arms_overhead(keypoints, min_score, aspect_ratio):
    """(both wrists above the top of the head, and also both elbows straight) for one frame.

    Args:
        keypoints: (17, 3) COCO-17 keypoints, x and y normalized to the frame (y down), or None.
        min_score: Joint confidence threshold.
        aspect_ratio: Frame width / height, so elbow angles are measured in real proportions.

    Returns:
        (above, straight); (False, False) when an arm or the head is not seen.
    """
    if keypoints is None or (keypoints[list(ARM_JOINTS), 2] < min_score).any():
        return False, False
    xy = keypoints[:, :2].astype(np.float64) * (aspect_ratio, 1.0)
    shoulder_y = (xy[LEFT_SHOULDER, 1] + xy[RIGHT_SHOULDER, 1]) / 2
    head_top = xy[NOSE, 1] - HEAD_TOP * (shoulder_y - xy[NOSE, 1])
    above = bool(xy[LEFT_WRIST, 1] < head_top and xy[RIGHT_WRIST, 1] < head_top)
    straight = above and min(_elbow_angle(xy, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                             _elbow_angle(xy, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)) >= STRAIGHT_ELBOW_DEG
    return above, straight


class OverheadHold:
    """Times how long the wrists stay above the head (item 8) and whether the arms are straight (item 7).

    Attributes:
        above: Wrists above the head in the latest frame.
        straight: Arms straight above the head in the latest frame.
        held_s: Length of the current hold (frozen during a gap shorter than GRACE_S, 0 after it).
        best_s: Longest hold so far.
    """

    def __init__(self):
        self.above = self.straight = False
        self.held_s = self.best_s = 0.0
        self._started = self._last_up = None

    def update(self, now, keypoints, min_score, aspect_ratio):
        """Add one frame; returns `held_s`."""
        self.above, self.straight = arms_overhead(keypoints, min_score, aspect_ratio)
        gap_over = self._last_up is None or now - self._last_up > GRACE_S
        if self.above:
            if gap_over:
                self._started = now
            self._last_up = now
            self.held_s = now - self._started
            self.best_s = max(self.best_s, self.held_s)
        elif gap_over:
            self.held_s = 0.0
        return self.held_s

    def panel_line(self):
        """(text, colour) for the camera overlay."""
        state = "straight" if self.straight else "up" if self.above else "down"
        return (f"arms {state}  held {self.held_s:.1f}s  best {self.best_s:.1f}s",
                GREEN if self.above else GREY)


if __name__ == "__main__":
    pose = np.zeros((17, 3), np.float32)
    pose[:, 2] = 1.0
    pose[NOSE, :2] = (0.5, 0.30)
    pose[[LEFT_SHOULDER, RIGHT_SHOULDER], :2] = (0.45, 0.40), (0.55, 0.40)
    pose[[LEFT_ELBOW, RIGHT_ELBOW], :2] = (0.45, 0.25), (0.55, 0.25)
    pose[[LEFT_WRIST, RIGHT_WRIST], :2] = (0.45, 0.10), (0.55, 0.10)  # straight up
    assert arms_overhead(pose, 0.5, 1.0) == (True, True)
    bent = pose.copy()
    bent[[LEFT_WRIST, RIGHT_WRIST], :2] = (0.50, 0.22), (0.50, 0.22)  # hands together, elbows bent
    assert arms_overhead(bent, 0.5, 1.0) == (True, False)  # wrists above the crown at y 0.25
    down = pose.copy()
    down[[LEFT_WRIST, RIGHT_WRIST], 1] = 0.6
    assert arms_overhead(down, 0.5, 1.0) == (False, False)
    unseen = pose.copy()
    unseen[NOSE, 2] = 0.0
    assert arms_overhead(unseen, 0.5, 1.0) == (False, False)

    hold = OverheadHold()
    for step in range(31):  # 3 s up at 10 FPS, one dropped frame in the middle
        hold.update(step / 10, down if step == 15 else pose, 0.5, 1.0)
    assert abs(hold.held_s - 3.0) < 1e-9, hold.held_s
    for step in range(31, 40):
        hold.update(step / 10, down, 0.5, 1.0)
    assert hold.held_s == 0.0 and abs(hold.best_s - 3.0) < 1e-9
    print("ok")
