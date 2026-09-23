import pytest

import robot.connect
from robot import preflight as reachy_preflight
from robot.preflight import PreflightResult


def test_resolve_host_default(monkeypatch):
    monkeypatch.delenv("REACHY_HOST", raising=False)
    assert robot.connect.resolve_host() == robot.connect.ROBOT_IP


def test_resolve_host_env(monkeypatch):
    monkeypatch.setenv("REACHY_HOST", "10.0.0.7")
    assert robot.connect.resolve_host() == "10.0.0.7"


def test_connect_robot_raises_when_not_ready(monkeypatch):
    calls = {}

    def fake_preflight(host=None, fix=False, needs=("motion",), **kwargs):
        calls.update(host=host, fix=fix, needs=set(needs))
        result = PreflightResult(host="robot")
        result.add("network", "failed", "unreachable")
        return result

    monkeypatch.setattr(reachy_preflight, "run_preflight", fake_preflight)
    with pytest.raises(robot.connect.RobotNotReady, match="UNREACHABLE"):
        with robot.connect.connect_robot(media=True):
            pass
    assert calls["fix"] is True
    assert calls["needs"] == {"motion", "media"}
