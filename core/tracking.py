"""Head tracking: which keypoint to follow, and how the image error becomes head and body angles."""

from typing import Optional, Tuple

import numpy as np

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER


def body_center(keypoints: Optional[np.ndarray], min_score: float) -> Optional[Tuple[float, float, str]]:
    """The point to follow: torso centre, else the shoulders, else the nose.

    Returns:
        (x, y, source) with source "torso", "shoulders" or "nose", or None if nothing is confident.
    """
    if keypoints is None:
        return None

    def confident(*indices):
        return [i for i in indices if keypoints[i, 2] >= min_score]

    shoulders, hips = confident(LEFT_SHOULDER, RIGHT_SHOULDER), confident(LEFT_HIP, RIGHT_HIP)
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


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Limit `value` to [minimum, maximum]."""
    return max(minimum, min(value, maximum))


def image_error_to_angles(x, y, max_yaw, max_pitch, deadzone) -> Tuple[float, float]:
    """Map a normalized image point (0..1, mirrored frame) to (yaw, pitch) in degrees.

    A point on the right means the person is on the robot's left (positive yaw); a point low in the
    image gives positive pitch (head looks down). Offsets smaller than `deadzone` are ignored.
    """
    error_x = clamp(x - 0.5, -0.5, 0.5)
    error_y = clamp(y - 0.5, -0.5, 0.5)
    if abs(error_x) < deadzone:
        error_x = 0.0
    if abs(error_y) < deadzone:
        error_y = 0.0
    return error_x * 2 * max_yaw, error_y * 2 * max_pitch


class HeadFollower:
    """Exponential smoothing toward the target; drifts back to centre when the target is lost (simulation)."""

    def __init__(self, smoothing: float, return_speed: float):
        """Set the smoothing factors (0..1, higher = faster)."""
        self.smoothing = smoothing
        self.return_speed = return_speed
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, target: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """Move toward `target` (yaw, pitch) or toward centre when None. Returns (yaw, pitch)."""
        alpha, (target_yaw, target_pitch) = (self.return_speed, (0.0, 0.0)) if target is None else (self.smoothing, target)
        self.yaw += alpha * (target_yaw - self.yaw)
        self.pitch += alpha * (target_pitch - self.pitch)
        return self.yaw, self.pitch


class CenteringFollower:
    """Integrating gaze controller for the robot, optionally helped by the rotating body.

    The image error feeds the gaze *rate*, so the person ends up centred (a position mapping settles
    with about 40% of the offset left). The head pose is absolute, so the body does not add to the gaze;
    it extends the head's +-60 degree range by following it.

    Attributes:
        gaze_yaw: Where the robot wants to look, in degrees.
        body_yaw: Body rotation to command, in degrees.
        yaw: Head yaw command (absolute), the gaze clamped to what the body allows.
        pitch: Head pitch command.
    """

    def __init__(self, gain, max_yaw, max_pitch, deadzone, return_speed, mirrored=True, body_gain=0.0,
                 max_head_offset=45.0, max_body_yaw=120.0):
        """Configure the controller.

        Args:
            gain: Degrees per second per unit of normalized error.
            max_yaw: Limit of the whole gaze (head + body).
            max_pitch: Pitch limit.
            deadzone: Ignored offset from the image centre.
            return_speed: Degrees per second back to centre when the person is lost.
            mirrored: False for the robot camera (not mirrored like a webcam preview).
            body_gain: How fast the body catches up with the gaze, per second (0 = body still).
            max_head_offset: How far the head may look away from the body (hardware limit 60).
            max_body_yaw: Body rotation limit.
        """
        self.gain = gain
        self.mirrored = mirrored
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.deadzone = deadzone
        self.return_speed = return_speed
        self.body_gain = body_gain
        self.max_head_offset = max_head_offset
        self.max_body_yaw = max_body_yaw
        self.gaze_yaw = self.body_yaw = self.yaw = self.pitch = 0.0

    def update(self, error, dt: float) -> Tuple[float, float]:
        """Integrate one step.

        Args:
            error: (x, y) offsets from the image centre in [-0.5, 0.5], or None when there is no target.
            dt: Seconds since the last update.

        Returns:
            Head (yaw, pitch) in the base frame; `body_yaw` holds the body rotation to command.
        """
        if error is None:
            self.gaze_yaw = _toward_zero(self.gaze_yaw, self.return_speed * dt)
            self.pitch = _toward_zero(self.pitch, self.return_speed * dt)
            self.body_yaw = _toward_zero(self.body_yaw, self.return_speed * dt)
        else:
            error_x, error_y = error
            if not self.mirrored:
                error_x = -error_x
            if abs(error_x) >= self.deadzone:
                self.gaze_yaw = clamp(self.gaze_yaw + self.gain * error_x * dt, -self.max_yaw, self.max_yaw)
            if abs(error_y) >= self.deadzone:
                self.pitch = clamp(self.pitch + self.gain * error_y * dt, -self.max_pitch, self.max_pitch)
            if self.body_gain:
                step = self.body_gain * (self.gaze_yaw - self.body_yaw) * dt
                self.body_yaw = clamp(self.body_yaw + step, -self.max_body_yaw, self.max_body_yaw)
        self.yaw = clamp(self.gaze_yaw, self.body_yaw - self.max_head_offset, self.body_yaw + self.max_head_offset)
        return self.yaw, self.pitch


def _toward_zero(value: float, step: float) -> float:
    if abs(value) <= step:
        return 0.0
    return value - step if value > 0 else value + step
