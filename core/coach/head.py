"""Who moves the head: head follow (`set_target` at 50 Hz) or a gesture (`goto_target`), never both."""

import time

from core.motion.antennas import NEUTRAL_ANTENNAS
from core.session.gestures import GestureStep

# Longer than the gap between two keyframes of a gesture, so follow does not jerk the head mid-gesture.
RESUME_AFTER_S = 0.3

THINKING_GESTURE = [
    GestureStep({"pitch": -6, "roll": 12, "yaw": 8, "degrees": True}, [0.5, -0.2], 0.8),
    GestureStep({"pitch": -6, "roll": -12, "yaw": -8, "degrees": True}, [-0.2, 0.5], 0.8),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.6),
]


class GestureMini:
    """The `mini` handed to gesture code: `goto_target` pauses the `TargetSender`, the rest passes through.

    Lets `SessionEngine` and `play_gesture_with_speech` run unchanged while head follow is on.
    """

    def __init__(self, mini, sender):
        self._mini = mini
        self._sender = sender
        self.closed = False

    def goto_target(self, *args, **kwargs):
        if self.closed:  # shutting down: a late gesture must not move a robot that is going to sleep
            return None
        self._sender.paused_until = float("inf")
        try:
            return self._mini.goto_target(*args, **kwargs)
        finally:
            self._sender.paused_until = time.perf_counter() + RESUME_AFTER_S

    def __getattr__(self, name):
        return getattr(self._mini, name)
