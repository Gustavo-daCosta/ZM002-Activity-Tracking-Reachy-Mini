import math

import pytest

from core.motion.reaction import AntennaWave


def test_angles_follow_sine_in_opposite_phase():
    reaction = AntennaWave(amplitude_deg=30, freq_hz=2.5, duration_s=1.2, cooldown_s=2.5)
    assert reaction.angles(0.0) is None
    assert reaction.trigger(10.0)
    right, left = reaction.angles(10.1)  # quarter period -> peak
    assert right == pytest.approx(math.radians(30))
    assert left == pytest.approx(-right)
    assert reaction.active(11.0)
    assert not reaction.active(11.25)
    assert reaction.angles(11.25) is None


def test_cooldown_blocks_retrigger():
    reaction = AntennaWave(duration_s=1.2, cooldown_s=2.5)
    assert reaction.trigger(0.0)
    assert not reaction.trigger(1.0)   # still waving
    assert not reaction.trigger(3.6)   # cooling down
    assert reaction.trigger(3.7)


def test_a_new_wave_is_accepted_right_after_the_cooldown():
    reaction = AntennaWave(duration_s=1.0, cooldown_s=0.5)
    assert reaction.trigger(0.0)
    assert not reaction.trigger(1.4)  # still inside duration + cooldown
    assert reaction.trigger(1.6)
