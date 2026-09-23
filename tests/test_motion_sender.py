"""The antenna wave is sent at its own rate, independent of the (slow) vision loop."""

import time

import pytest

from core.motion.reaction import AntennaWave, NEUTRAL_ANTENNAS
from core.motion.sender import TargetSender


class FakeMini:
    def __init__(self):
        self.calls = []

    def set_target(self, head=None, antennas=None, body_yaw=None):
        self.calls.append((head, antennas, body_yaw))


def test_sends_at_its_own_rate_while_the_caller_is_busy():
    mini, reaction = FakeMini(), AntennaWave()
    with TargetSender(mini, reaction, rate_hz=200):
        time.sleep(0.2)  # a single slow "vision frame"

    assert len(mini.calls) >= 10


def test_neutral_antennas_when_no_wave_is_running():
    mini, reaction = FakeMini(), AntennaWave()
    with TargetSender(mini, reaction, rate_hz=200):
        time.sleep(0.05)

    assert all(antennas == list(NEUTRAL_ANTENNAS) for _, antennas, _ in mini.calls)


def test_wave_makes_the_antennas_move_between_sends():
    mini, reaction = FakeMini(), AntennaWave(amplitude_deg=20, freq_hz=1.5)
    with TargetSender(mini, reaction, rate_hz=200):
        reaction.trigger(time.perf_counter())
        time.sleep(0.3)

    right = [antennas[0] for _, antennas, _ in mini.calls]
    assert max(right) - min(right) > 0.05  # radians: clearly moving, not one step


def test_forwards_the_latest_head_pose():
    mini, reaction = FakeMini(), AntennaWave()
    with TargetSender(mini, reaction, rate_hz=200) as sender:
        sender.head = "pose-a"
        time.sleep(0.05)

    assert mini.calls[-1][0] == "pose-a"


def test_stops_sending_after_the_context_exits():
    mini, reaction = FakeMini(), AntennaWave()
    with TargetSender(mini, reaction, rate_hz=200):
        time.sleep(0.05)
    count = len(mini.calls)
    time.sleep(0.05)

    assert len(mini.calls) == count


def test_body_yaw_is_forwarded_in_radians():
    import math

    mini, reaction = FakeMini(), AntennaWave()
    with TargetSender(mini, reaction, rate_hz=200) as sender:
        sender.body_yaw_deg = 30.0
        time.sleep(0.05)

    assert mini.calls[-1][2] == pytest.approx(math.radians(30.0))


def test_body_yaw_is_none_until_the_caller_sets_it():
    mini, reaction = FakeMini(), AntennaWave()
    with TargetSender(mini, reaction, rate_hz=200):
        time.sleep(0.03)

    assert all(body is None for _, _, body in mini.calls)
