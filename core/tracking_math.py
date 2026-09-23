"""Pure math for webcam head tracking: image position -> head angles, plus smoothing.

Kept free of OpenCV/MediaPipe/SDK imports so it can be unit tested without hardware.
"""

from typing import Optional, Tuple


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def image_error_to_angles(
    x: float, y: float, max_yaw: float, max_pitch: float, deadzone: float
) -> Tuple[float, float]:
    """Map a normalized image point (0..1, mirrored frame) to (yaw, pitch) in degrees.

    The frame is mirrored, so a point on the right means the person is on the robot's left:
    positive yaw (head turns left). A point low in the image gives positive pitch (head looks
    down). Offsets smaller than `deadzone` from the center are ignored.
    """
    error_x = clamp(x - 0.5, -0.5, 0.5)
    error_y = clamp(y - 0.5, -0.5, 0.5)
    if abs(error_x) < deadzone:
        error_x = 0.0
    if abs(error_y) < deadzone:
        error_y = 0.0
    return error_x * 2 * max_yaw, error_y * 2 * max_pitch


class HeadFollower:
    """Exponential smoothing toward the target; drifts back to center when the target is lost."""

    def __init__(self, smoothing: float, return_speed: float):
        self.smoothing = smoothing
        self.return_speed = return_speed
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, target: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        if target is None:
            alpha, (target_yaw, target_pitch) = self.return_speed, (0.0, 0.0)
        else:
            alpha, (target_yaw, target_pitch) = self.smoothing, target
        self.yaw += alpha * (target_yaw - self.yaw)
        self.pitch += alpha * (target_pitch - self.pitch)
        return self.yaw, self.pitch


class CenteringFollower:
    """Integrating gaze controller, optionally helped by the rotating body.

    `HeadFollower` maps the image error straight to an angle, so turning the head shrinks the error and the
    motion settles with the person still off-center (measured on the robot: roughly 40 % of the offset is
    corrected). Here the error feeds the *rate*: the gaze keeps moving at `gain` degrees per second per unit
    of normalized error until the person is inside the deadzone, which is what actually centers them.

    The head pose is **absolute** (base frame): measured on the robot, rotating the body makes the IK
    counter-rotate the head and the camera stays where it was. So the body does not add to the gaze - it
    *extends its range*, because the head can only reach +-60 deg relative to the body. With `body_gain > 0`
    the body follows the gaze so the head keeps working inside `max_head_offset` of it; body yaw is +-155 deg
    mechanical.
    """

    def __init__(self, gain: float, max_yaw: float, max_pitch: float, deadzone: float, return_speed: float,
                 mirrored: bool = True, body_gain: float = 0.0, max_head_offset: float = 45.0,
                 max_body_yaw: float = 120.0):
        self.gain = gain
        # The robot camera shows the world from the robot's point of view (not mirrored like a webcam
        # preview): a person on the right of the image is on the robot's right, so the yaw is inverted.
        self.mirrored = mirrored
        self.max_yaw = max_yaw          # limit of the whole gaze (head + body)
        self.max_pitch = max_pitch
        self.deadzone = deadzone
        self.return_speed = return_speed  # degrees per second back to center when the person is lost
        self.body_gain = body_gain      # how fast the body catches up with the gaze (per second)
        self.max_head_offset = max_head_offset  # how far the head may look away from the body (hw limit 60)
        self.max_body_yaw = max_body_yaw
        self.gaze_yaw = 0.0   # where the robot wants to look, in degrees
        self.body_yaw = 0.0   # body rotation, which carries the head's reachable range
        self.yaw = 0.0        # head command (absolute), the gaze clamped to what the body allows
        self.pitch = 0.0

    def update(self, error, dt: float) -> Tuple[float, float]:
        """error: (x, y) offsets from the image center in [-0.5, 0.5], or None when there is no target.

        Returns the head (yaw, pitch) in the base frame; `body_yaw` is the body rotation to command.
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
        # The head looks where the gaze wants, as far as its offset from the body allows.
        self.yaw = clamp(self.gaze_yaw, self.body_yaw - self.max_head_offset, self.body_yaw + self.max_head_offset)
        return self.yaw, self.pitch


def _toward_zero(value: float, step: float) -> float:
    if abs(value) <= step:
        return 0.0
    return value - step if value > 0 else value + step
