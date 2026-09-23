import numpy as np
import pytest

from core.body_center import body_center


def keypoints(**points):
    """Build (17, 3) keypoints; unspecified joints have score 0."""
    index = {"nose": 0, "ls": 5, "rs": 6, "lh": 11, "rh": 12, "lw": 9}
    kps = np.zeros((17, 3), np.float32)
    for name, value in points.items():
        kps[index[name]] = value
    return kps


def assert_center(result, x, y, source):
    assert result is not None
    assert result[2] == source
    assert result[:2] == pytest.approx((x, y))


def test_none_keypoints():
    assert body_center(None, 0.5) is None


def test_full_torso_is_mean_of_shoulders_and_hips():
    kps = keypoints(ls=(0.4, 0.3, 0.9), rs=(0.6, 0.3, 0.9), lh=(0.45, 0.7, 0.9), rh=(0.55, 0.7, 0.9))
    assert_center(body_center(kps, 0.5), 0.5, 0.5, "torso")


def test_one_shoulder_and_one_hip_is_still_torso():
    kps = keypoints(ls=(0.4, 0.2, 0.9), rh=(0.6, 0.6, 0.9), rs=(0.9, 0.9, 0.1))
    assert_center(body_center(kps, 0.5), 0.5, 0.4, "torso")


def test_hips_out_of_frame_falls_back_to_shoulders():
    kps = keypoints(ls=(0.4, 0.8, 0.9), rs=(0.6, 0.8, 0.9), lh=(0.45, 1.4, 0.01), rh=(0.55, 1.4, 0.01))
    assert_center(body_center(kps, 0.5), 0.5, 0.8, "shoulders")


def test_only_nose():
    kps = keypoints(nose=(0.3, 0.2, 0.8), ls=(0.1, 0.1, 0.2))
    assert_center(body_center(kps, 0.5), 0.3, 0.2, "nose")


def test_nothing_confident_other_joints_ignored():
    kps = keypoints(nose=(0.3, 0.2, 0.4), lw=(0.5, 0.5, 0.99))
    assert body_center(kps, 0.5) is None
