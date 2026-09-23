from core.motion import OTHER, PAUSE, WAVE
from training.record import build_schedule, phase_at


def test_schedule_alternates_rounds_with_pauses():
    schedule = build_schedule(rounds=2, round_s=4.0, pause_s=2.0, prep_s=3.0)
    assert [(p.label, p.round_id) for p in schedule] == [
        (PAUSE, -1), (WAVE, 0), (PAUSE, -1), (OTHER, 1), (PAUSE, -1), (WAVE, 2), (PAUSE, -1), (OTHER, 3),
    ]
    assert schedule[1].start == 3.0 and schedule[1].end == 7.0
    assert schedule[-1].end == 25.0
    assert "NOT WAVE" in schedule[3].text and " - " in schedule[3].text  # hint for non-wave rounds


def test_phase_at():
    schedule = build_schedule(rounds=2, round_s=4.0, pause_s=2.0, prep_s=3.0)
    assert phase_at(schedule, 0.0).label == PAUSE
    assert phase_at(schedule, 3.5).round_id == 0
    assert phase_at(schedule, 8.0).label == PAUSE
    assert phase_at(schedule, 24.9).round_id == 3
    assert phase_at(schedule, 25.0) is None
