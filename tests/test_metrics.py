import pytest

from core.metrics import StageStats, format_summary


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_window_summary_updates_after_window():
    clock = FakeClock()
    stats = StageStats(window_s=1.0, clock=clock)
    assert stats.window_summary() == {"fps": 0.0, "ms": {}}

    for i in range(10):
        clock.now = 0.1 * (i + 1)
        stats.add_frame({"detector": 10.0, "pose": 20.0}, has_target=True)

    panel = stats.window_summary()
    assert panel["fps"] == pytest.approx(10.0)
    assert panel["ms"] == pytest.approx({"detector": 10.0, "pose": 20.0, "total": 30.0})


def test_final_summary_mean_p95_fps_and_person_pct():
    clock = FakeClock()
    stats = StageStats(clock=clock)
    for i in range(1, 21):  # 20 frames over 2 s, pose 1..20 ms, target in 15 frames
        clock.now = i * 0.1
        stats.add_frame({"pose": float(i)}, has_target=i <= 15)

    summary = stats.final_summary()
    assert summary["frames"] == 20
    assert summary["fps"] == pytest.approx(10.0)
    assert summary["person_pct"] == pytest.approx(75.0)
    assert summary["stages"]["pose"]["mean"] == pytest.approx(10.5)
    assert summary["stages"]["pose"]["p95"] == pytest.approx(19.05)
    assert summary["stages"]["total"]["mean"] == pytest.approx(10.5)


def test_frames_without_timings_do_not_add_total():
    stats = StageStats(clock=FakeClock())
    stats.add_frame({}, has_target=False)
    assert stats.final_summary()["stages"] == {}


def test_empty_final_summary():
    summary = StageStats(clock=FakeClock()).final_summary()
    assert summary == {"frames": 0, "fps": 0.0, "person_pct": 0.0, "stages": {}}


def test_format_summary_mentions_model_and_stages():
    text = format_summary(
        "vitpose-s",
        {"frames": 3, "fps": 12.5, "person_pct": 66.7,
         "stages": {"detector": {"mean": 30.0, "p95": 35.0}, "total": {"mean": 70.0, "p95": 80.0}}},
    )
    assert "vitpose-s" in text and "detector" in text and "12.5" in text and "p95" in text
