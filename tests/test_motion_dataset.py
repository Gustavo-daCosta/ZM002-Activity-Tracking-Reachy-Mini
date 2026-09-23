import numpy as np
import pytest

from motion_helpers import synthetic_session
from training.dataset import list_sessions, load_session, make_windows, save_session


def test_session_round_trip(tmp_path):
    session = synthetic_session(round_kinds=("wave", "still"), round_s=2.0)
    path = tmp_path / "wave" / "session_20260916_120000.npz"
    save_session(path, session["t"], session["keypoints"], session["labels"], session["rounds"], "blazepose-lite", 16 / 9)

    loaded = load_session(path)
    assert list_sessions(tmp_path / "wave") == [path]
    assert np.array_equal(loaded["t"], session["t"])
    assert np.array_equal(loaded["keypoints"], session["keypoints"])
    assert np.array_equal(loaded["labels"], session["labels"])
    assert np.array_equal(loaded["rounds"], session["rounds"])
    assert loaded["pose_model"] == "blazepose-lite"
    assert loaded["aspect_ratio"] == pytest.approx(16 / 9)


def test_windows_stay_inside_rounds():
    session = synthetic_session(round_kinds=("wave", "still"), round_s=3.0, pause_s=1.0)
    windows = make_windows(session)

    wave_round = [features for features, label, round_id in windows if round_id == 0]
    still_round = [features for features, label, round_id in windows if round_id == 1]
    assert len(wave_round) >= 5 and len(still_round) >= 5
    assert {label for _, label, round_id in windows if round_id == 0} == {1}
    assert {label for _, label, round_id in windows if round_id == 1} == {0}
    assert max(f["reversals"] for f in wave_round) >= 3  # early windows of a slow wave may see fewer
    assert all(f["reversals"] == 0 for f in still_round)  # a window mixing in wave frames would reverse
    assert all(round_id >= 0 for _, _, round_id in windows)  # pause frames never form windows
