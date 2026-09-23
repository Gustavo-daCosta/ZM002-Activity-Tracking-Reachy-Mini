import pytest

from core.tracking_math import HeadFollower, clamp, image_error_to_angles


def test_clamp():
    assert clamp(5, -1, 1) == 1
    assert clamp(-5, -1, 1) == -1
    assert clamp(0.3, -1, 1) == 0.3


def test_center_gives_zero_angles():
    assert image_error_to_angles(0.5, 0.5, max_yaw=40, max_pitch=30, deadzone=0.03) == (0.0, 0.0)


def test_deadzone_ignores_small_offsets():
    assert image_error_to_angles(0.52, 0.48, max_yaw=40, max_pitch=30, deadzone=0.03) == (0.0, 0.0)


def test_right_and_down_give_positive_yaw_and_pitch():
    # Mirrored frame: nose on the right edge / bottom edge -> full positive yaw / pitch.
    yaw, pitch = image_error_to_angles(1.0, 1.0, max_yaw=40, max_pitch=30, deadzone=0.0)
    assert yaw == pytest.approx(40)
    assert pitch == pytest.approx(30)


def test_out_of_frame_is_clamped():
    yaw, pitch = image_error_to_angles(-0.4, 1.7, max_yaw=40, max_pitch=30, deadzone=0.0)
    assert yaw == pytest.approx(-40)
    assert pitch == pytest.approx(30)


def test_follower_smooths_toward_target():
    f = HeadFollower(smoothing=0.5, return_speed=0.1)
    assert f.update((10.0, -10.0)) == pytest.approx((5.0, -5.0))
    assert f.update((10.0, -10.0)) == pytest.approx((7.5, -7.5))


def test_follower_drifts_back_to_center_when_target_lost():
    f = HeadFollower(smoothing=1.0, return_speed=0.5)
    f.update((20.0, 10.0))
    assert f.update(None) == pytest.approx((10.0, 5.0))
    assert f.update(None) == pytest.approx((5.0, 2.5))

