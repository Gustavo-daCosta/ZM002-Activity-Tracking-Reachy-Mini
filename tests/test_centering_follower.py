"""Integrating follower: keeps the person centered instead of settling for a fraction of the error."""

import pytest

from core.tracking_math import CenteringFollower


def follower(**kwargs):
    defaults = dict(gain=120.0, max_yaw=35.0, max_pitch=20.0, deadzone=0.05, return_speed=10.0)
    return CenteringFollower(**{**defaults, **kwargs})


def test_keeps_turning_while_the_error_stays():
    head = follower()
    first = head.update((0.2, 0.0), 0.1)[0]
    second = head.update((0.2, 0.0), 0.1)[0]

    assert 0 < first < second  # a proportional controller would stop at the same angle


def test_a_centered_target_holds_the_angle():
    head = follower()
    head.update((0.3, 0.0), 0.5)
    held = head.update((0.0, 0.0), 0.5)

    assert held == head.update((0.0, 0.0), 0.5)


def test_deadzone_ignores_small_offsets():
    head = follower()
    assert head.update((0.03, -0.02), 0.5) == (0.0, 0.0)


def test_angles_are_clamped_to_the_limits():
    head = follower()
    for _ in range(50):
        yaw, pitch = head.update((0.5, 0.5), 0.2)

    assert yaw == pytest.approx(35.0)
    assert pitch == pytest.approx(20.0)


def test_sign_follows_the_error():
    head = follower()
    yaw, pitch = head.update((-0.4, -0.4), 0.2)

    assert yaw < 0 and pitch < 0


def test_lost_target_drifts_back_to_center():
    head = follower()
    head.update((0.5, 0.5), 1.0)
    before = head.update(None, 0.1)
    after = head.update(None, 0.1)

    assert abs(after[0]) < abs(before[0])
    for _ in range(200):
        yaw, pitch = head.update(None, 0.1)
    assert (yaw, pitch) == (0.0, 0.0)


def test_unmirrored_camera_turns_the_head_toward_the_person():
    """Robot camera: a person on the right of the image needs a negative (right) yaw."""
    head = follower(mirrored=False)
    yaw, pitch = head.update((0.3, 0.3), 0.2)

    assert yaw < 0
    assert pitch > 0  # pitch is unaffected by mirroring


def assisted(**kwargs):
    """Follower whose body yaw takes over the large angles, like a neck plus a waist."""
    defaults = dict(gain=120.0, max_yaw=90.0, max_pitch=20.0, deadzone=0.05, return_speed=20.0,
                    body_gain=2.0, max_head_offset=25.0, max_body_yaw=120.0)
    return CenteringFollower(**{**defaults, **kwargs})


def test_body_rotates_to_let_the_head_reach_further():
    head = assisted()
    for _ in range(20):
        yaw, _ = head.update((0.4, 0.0), 0.1)

    assert head.gaze_yaw > 40.0  # the gaze went past what the head alone can reach from a still body
    assert head.body_yaw > 20.0  # because the body rotated
    assert yaw == pytest.approx(head.gaze_yaw)  # the head command is absolute: it points at the gaze
    assert abs(yaw - head.body_yaw) <= 25.0  # within the head's offset limit


def test_head_leads_and_the_body_catches_up():
    head = assisted()
    first_head, _ = head.update((0.4, 0.0), 0.1)
    first_body = head.body_yaw
    for _ in range(15):
        head.update((0.4, 0.0), 0.1)

    assert first_head > first_body  # the head moves first
    assert first_body < head.body_yaw  # and the body keeps catching up


def test_head_command_is_capped_by_the_offset_from_the_body():
    head = assisted(body_gain=0.0)  # body stays at 0
    for _ in range(30):
        yaw, _ = head.update((0.5, 0.0), 0.2)

    assert head.gaze_yaw == pytest.approx(90.0)  # the gaze saturates at its own limit
    assert yaw == pytest.approx(25.0)  # but the head cannot look further than its offset from the body


def test_body_yaw_is_clamped():
    head = assisted(max_body_yaw=15.0)
    for _ in range(50):
        head.update((0.5, 0.0), 0.2)

    assert head.body_yaw == pytest.approx(15.0)


def test_losing_the_person_returns_head_and_body_to_center():
    head = assisted()
    for _ in range(20):
        head.update((0.4, 0.2), 0.1)
    for _ in range(400):
        yaw, pitch = head.update(None, 0.1)

    assert (yaw, pitch) == (0.0, 0.0)
    assert head.body_yaw == 0.0
    assert head.gaze_yaw == 0.0


def test_body_stays_still_when_the_assist_is_off():
    head = follower()  # body_gain defaults to 0
    for _ in range(20):
        head.update((0.4, 0.0), 0.1)

    assert head.body_yaw == 0.0
