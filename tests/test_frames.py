"""Frame sources used by the simulation (webcam) and by the robot scripts (SDK camera)."""

import numpy as np
import pytest

from core.frames import RobotCameraSource, downscale


class FakeMedia:
    def __init__(self, frames):
        self.frames = list(frames)

    def get_frame(self):
        return self.frames.pop(0) if self.frames else None


class FakeMini:
    def __init__(self, frames):
        self.media = FakeMedia(frames)


def test_robot_source_returns_a_writable_copy():
    frame = np.zeros((4, 6, 3), np.uint8)
    frame.flags.writeable = False  # SDK frames are read-only
    source = RobotCameraSource(FakeMini([frame]))

    out = source.read()

    assert out is not None and out.flags.writeable
    out[0, 0] = 255  # would raise on the original


def test_robot_source_returns_none_while_frames_are_missing():
    source = RobotCameraSource(FakeMini([None]))
    assert source.read() is None


def test_downscale_keeps_the_aspect_ratio():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = downscale(frame, 640)
    assert out.shape[:2] == (360, 640)


def test_downscale_never_upscales():
    frame = np.zeros((480, 640, 3), np.uint8)
    assert downscale(frame, 1280) is frame


@pytest.mark.parametrize("width", [0, -1])
def test_downscale_ignores_a_non_positive_width(width):
    frame = np.zeros((480, 640, 3), np.uint8)
    assert downscale(frame, width) is frame
