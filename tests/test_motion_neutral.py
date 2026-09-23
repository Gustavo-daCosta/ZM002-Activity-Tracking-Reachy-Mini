"""Antenna angles are sent around the robot's neutral pose, not around zero."""

import pytest

from core.motion.reaction import NEUTRAL_ANTENNAS, antennas_with_neutral


def test_no_wave_keeps_the_neutral_pose():
    assert antennas_with_neutral(None) == list(NEUTRAL_ANTENNAS)


def test_wave_is_added_around_neutral():
    angles = antennas_with_neutral([0.2, -0.2])
    assert angles == pytest.approx([NEUTRAL_ANTENNAS[0] + 0.2, NEUTRAL_ANTENNAS[1] - 0.2])
