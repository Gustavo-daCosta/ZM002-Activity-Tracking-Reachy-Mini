"""The recognition entry points, as far as they can be checked without a camera or a robot.

Not covered here (needs hardware): the camera loop itself, the robot connection and the antenna motion.
"""

import signal

import numpy as np
import pytest

from robot.apps import common
from robot.apps.action_recognition import parse_args
from robot.apps.common import install_signal_handlers
from core.motion.actions.detector import DEFAULT_ACTION_MODEL
from core.motion.actions.live import ActionMonitor
from core.motion.reaction import NEUTRAL_ANTENNAS, antennas_with_neutral


def test_defaults_are_the_robot_path():
    args = parse_args([])
    assert args.host == "localhost"
    assert args.model == "movenet-tflite"  # mediapipe aborts on the robot's CPU
    assert args.classifier is None  # -> the shipped .npz
    assert args.window is False and args.no_sleep is False


def test_overrides():
    args = parse_args(["--host", "192.168.1.5", "--window", "--seconds", "30", "--no-sleep",
                       "--classifier", "some/model.npz", "--width", "480"])
    assert (args.host, args.window, args.seconds, args.no_sleep) == ("192.168.1.5", True, 30.0, True)
    assert args.classifier == "some/model.npz"
    assert args.width == 480


def test_wave_app_shares_the_options():
    from robot.apps.wave_antennas import parse_args as wave_parse_args

    args = wave_parse_args(["--wave-trigger", "rules", "--amplitude", "10"])
    assert args.model == "movenet-tflite" and args.wave_trigger == "rules" and args.amplitude == 10.0


def _monitor():
    if not DEFAULT_ACTION_MODEL.exists():
        pytest.skip("no shipped action model")
    return ActionMonitor(4 / 3, min_score=0.5)


def test_monitor_requires_an_aspect_ratio():
    with pytest.raises(TypeError):
        ActionMonitor()  # no default, deliberately


def test_monitor_antennas_include_the_neutral_pose_exactly_once():
    monitor = _monitor()
    assert monitor.antennas(5.0) == pytest.approx(list(NEUTRAL_ANTENNAS))
    assert monitor.reactions["wave"].trigger(100.0)
    for now in (100.0, 100.1, 100.35, 100.9, 101.32, 102.0, 110.0):
        expected = antennas_with_neutral(monitor.reactions["wave"].angles(now))
        assert monitor.antennas(now) == pytest.approx(expected)


def test_signal_handler_raises_keyboard_interrupt():
    handler = install_signal_handlers()
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_handler_is_installed_for_both(signum):
    """SIGTERM: stopped over SSH. SIGHUP: the SSH connection dropped. Both must run the cleanup."""
    previous = signal.getsignal(signum)
    try:
        handler = install_signal_handlers()
        assert signal.getsignal(signum) is handler
    finally:
        signal.signal(signum, previous)


def test_body_tracking_action_flag():
    from sim.body_tracking import parse_args as sim_parse_args

    assert sim_parse_args([]).detect_actions is False
    args = sim_parse_args(["--detect-actions"])
    assert args.detect_actions is True and args.detect_wave is False
    # The camera is chosen by name: OpenCV indices move when an iPhone joins as a Continuity Camera.
    assert not str(args.camera).isdigit()


class FakeMini:
    """Records the calls the cleanup path is supposed to make, so an exit can be checked without a robot."""

    def __init__(self):
        self.slept = 0
        self.targets = 0

    def goto_sleep(self):
        self.slept += 1

    def goto_target(self, **kwargs):
        self.targets += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeBackend:
    name = "fake"
    default_min_score = 0.3

    def __init__(self):
        self.closed = 0

    def infer(self, frame):  # pragma: no cover - the loop is never reached in these tests
        raise AssertionError("the vision loop should not run in these tests")

    def close(self):
        self.closed += 1


def _run_main(monkeypatch, argv, on_frames, monitor_factory=None):
    """Run `main()` with the robot, the camera and the pose model faked out. Returns (FakeMini, module).

    `on_frames` replaces `wait_for_frames` (return None, a frame, or raise); `monitor_factory` replaces the
    ActionMonitor constructor, which is how the interrupt window between the frame wait and the loop is
    exercised.
    """
    from core.motion.actions import live as live_module
    from robot.apps import action_recognition as module

    mini = FakeMini()
    monkeypatch.setattr(common, "create_backend", lambda name: FakeBackend())
    monkeypatch.setattr(common, "connect", lambda host: mini)
    monkeypatch.setattr(common, "RobotCameraSource", lambda mini_: object())
    monkeypatch.setattr(common, "downscale", lambda frame, width: frame)
    monkeypatch.setattr(common, "wait_for_frames", lambda source, **kw: on_frames())
    if monitor_factory is not None:
        monkeypatch.setattr(live_module, "ActionMonitor", monitor_factory)
    monkeypatch.setattr("sys.argv", ["action_recognition", *argv])
    return mini, module


@pytest.mark.parametrize("no_sleep, expected", [([], 1), (["--no-sleep"], 0)])
def test_no_camera_frames_still_puts_the_robot_to_sleep(monkeypatch, no_sleep, expected):
    mini, module = _run_main(monkeypatch, no_sleep, on_frames=lambda: None)
    with pytest.raises(SystemExit):
        module.main()
    assert mini.slept == expected


@pytest.mark.parametrize("no_sleep, expected", [([], 1), (["--no-sleep"], 0)])
def test_interrupt_while_waiting_for_frames_puts_the_robot_to_sleep(monkeypatch, no_sleep, expected):
    def interrupt():
        raise KeyboardInterrupt

    mini, module = _run_main(monkeypatch, no_sleep, on_frames=interrupt)
    module.main()  # the interrupt is handled, not propagated
    assert mini.slept == expected


@pytest.mark.parametrize("no_sleep, expected", [([], 1), (["--no-sleep"], 0)])
def test_interrupt_between_the_frame_wait_and_the_loop_puts_the_robot_to_sleep(monkeypatch, no_sleep, expected):
    """A SIGHUP arriving while the ActionMonitor is built must still reach the cleanup."""

    def interrupting_monitor(*args, **kwargs):
        raise KeyboardInterrupt

    mini, module = _run_main(monkeypatch, no_sleep, on_frames=lambda: np.zeros((4, 4, 3), np.uint8),
                             monitor_factory=interrupting_monitor)
    module.main()
    assert mini.slept == expected
