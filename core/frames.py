"""Frame sources for the pose models: the robot camera through the SDK (see core.vision for webcams)."""

import cv2


class RobotCameraSource:
    """Frames from the robot camera through the SDK (BGR 1280x720), copied because SDK frames are read-only."""

    def __init__(self, mini):
        self._mini = mini

    def read(self):
        """The latest frame as a writable copy, or None when no frame is available."""
        frame = self._mini.media.get_frame()
        return None if frame is None else frame.copy()


def downscale(frame, width):
    """Shrink a frame to `width` keeping its aspect ratio (never upscales)."""
    if width <= 0 or frame.shape[1] <= width:
        return frame
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
