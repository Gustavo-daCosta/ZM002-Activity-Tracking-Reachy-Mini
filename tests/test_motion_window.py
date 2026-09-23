import numpy as np
import pytest

from motion_helpers import pose
from core.motion.window import KeypointWindow, normalize_to_shoulders


def test_normalize_to_shoulders_units_and_origin():
    kps = pose(wrist_xy=(0.7, 0.4))  # shoulders (0.4, 0.5) and (0.6, 0.5): width 0.2
    norm = normalize_to_shoulders(kps, min_score=0.5, aspect_ratio=1.0)
    assert norm[5, :2] == pytest.approx([-0.5, 0.0])
    assert norm[10, :2] == pytest.approx([1.0, -0.5])
    assert norm[10, 2] == pytest.approx(0.9)


def test_normalize_corrects_aspect_ratio():
    kps = pose(wrist_xy=(0.5, 0.3))
    norm = normalize_to_shoulders(kps, min_score=0.5, aspect_ratio=2.0)  # shoulder width 0.4 after x * 2
    assert norm[10, :2] == pytest.approx([0.0, -0.5])


def test_missing_shoulders_is_invalid():
    kps = pose()
    kps[5, 2] = 0.1
    assert normalize_to_shoulders(kps, 0.5, 1.0) is None
    assert normalize_to_shoulders(None, 0.5, 1.0) is None


def test_window_drops_old_frames_and_counts_invalid():
    window = KeypointWindow(duration_s=1.0, min_score=0.5)
    for i in range(30):  # 1.5 s at 20 Hz, every 4th frame has no person
        window.add(i / 20, None if i % 4 == 0 else pose())
    times, kps = window.frames()
    assert window.span() == pytest.approx(1.0)
    assert times[0] >= times[-1] - 1.0 - 1e-9
    invalid = np.isnan(kps[:, 0, 0])
    assert 0 < invalid.sum() < len(times)
    assert window.valid_frac() == pytest.approx(1 - invalid.mean())

    window.clear()
    assert window.span() == 0.0 and window.valid_frac() == 0.0
