import pytest

from motion_helpers import motion_frames
from core.motion.live import WaveMonitor


def run(monitor, kind):
    frames = motion_frames(kind, duration_s=2.0)
    for t, keypoints in frames:
        monitor.update(t, keypoints)
    return frames[-1][0]


def test_rules_trigger_the_antenna_wave_once(tmp_path):
    monitor = WaveMonitor(trigger="auto", model_path=tmp_path / "missing.joblib", aspect_ratio=1.0)
    assert monitor.trigger_name == "rules"
    last_t = run(monitor, "wave")
    assert monitor.triggers == {"rules": 1, "classifier": 0}
    assert monitor.detections["rules"].is_wave
    assert monitor.detections["classifier"] is None
    assert monitor.reaction.active(last_t)
    texts = [text for text, _ in monitor.panel_lines(last_t)]
    assert any(text.startswith("rules WAVE") for text in texts)
    assert any("no model" in text for text in texts)
    assert any(text.startswith("WAVE!") for text in texts)


def test_still_person_does_not_trigger(tmp_path):
    monitor = WaveMonitor(trigger="rules", model_path=tmp_path / "missing.joblib", aspect_ratio=1.0)
    last_t = run(monitor, "still")
    assert monitor.triggers["rules"] == 0
    assert monitor.antennas(last_t) == [0.0, 0.0]


def test_classifier_trigger_requires_a_model(tmp_path):
    with pytest.raises(ValueError, match="training.train"):
        WaveMonitor(trigger="classifier", model_path=tmp_path / "missing.joblib")


def test_reactions_only_count_waves_the_antennas_actually_answered(tmp_path):
    """Detections during the antenna cooldown are counted as detections, not as reactions."""
    monitor = WaveMonitor(trigger="auto", model_path=tmp_path / "missing.joblib", aspect_ratio=1.0)
    monitor.reaction.duration_s, monitor.reaction.cooldown_s = 0.5, 10.0

    for start, kind in ((0.0, "wave"), (2.0, "still"), (4.0, "wave")):
        for timestamp, keypoints in motion_frames(kind, duration_s=2.0, start=start):
            monitor.update(timestamp, keypoints)

    assert monitor.triggers["rules"] == 2
    assert monitor.reactions == 1


def test_a_sustained_wave_is_answered_again_after_the_cooldown(tmp_path):
    """Waving on and on must keep the antennas going, not stop at the first rising edge."""
    monitor = WaveMonitor(trigger="auto", model_path=tmp_path / "missing.joblib", aspect_ratio=1.0)
    monitor.reaction.duration_s, monitor.reaction.cooldown_s = 0.5, 0.2

    for start in (0.0, 2.0, 4.0, 6.0):
        for timestamp, keypoints in motion_frames("wave", duration_s=2.0, start=start):
            monitor.update(timestamp, keypoints)

    assert monitor.triggers["rules"] == 1  # one continuous detection
    assert monitor.reactions >= 3  # but the antennas answered several times


def feed(monitor, kind, start, duration_s=2.0):
    for timestamp, keypoints in motion_frames(kind, duration_s=duration_s, start=start):
        monitor.update(timestamp, keypoints)
    return start + duration_s


def rules_monitor(tmp_path, duration_s=0.5, cooldown_s=0.3):
    monitor = WaveMonitor(trigger="auto", model_path=tmp_path / "missing.joblib", aspect_ratio=1.0)
    monitor.reaction.duration_s, monitor.reaction.cooldown_s = duration_s, cooldown_s
    return monitor


def test_one_wave_is_answered_once(tmp_path):
    """The 1.5 s window still remembers the wave after it ends: that must not count as a second wave."""
    monitor = rules_monitor(tmp_path)

    clock = feed(monitor, "wave", 0.0, duration_s=2.0)
    feed(monitor, "still", clock, duration_s=4.0)

    assert monitor.reactions == 1


def test_two_separate_waves_are_answered_twice(tmp_path):
    monitor = rules_monitor(tmp_path)

    clock = feed(monitor, "wave", 0.0, duration_s=2.0)
    clock = feed(monitor, "still", clock, duration_s=3.0)
    feed(monitor, "wave", clock, duration_s=2.0)

    assert monitor.reactions == 2


def test_waving_without_stopping_keeps_being_answered(tmp_path):
    monitor = rules_monitor(tmp_path)

    feed(monitor, "wave", 0.0, duration_s=8.0)

    assert monitor.reactions >= 3
