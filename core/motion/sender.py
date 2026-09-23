"""Sends targets to the robot at a steady rate, independent of the vision loop.

Pose inference costs ~110 ms on the robot (~6.5 FPS), so sampling the antenna wave inside that loop gives
about 4 points per cycle and the motion looks stepped. This thread samples the same `AntennaWave` at the
daemon's own control rate (50 Hz) while the vision loop only decides *when* a wave starts.
"""

import math
import threading
import time

from core.motion.reaction import antennas_with_neutral


class TargetSender:
    """Context manager: while open, `mini.set_target` is called `rate_hz` times per second."""

    def __init__(self, mini, reaction, rate_hz=50.0, head=None, body_yaw_deg=None):
        self._mini = mini
        self._reaction = reaction
        self._interval = 1.0 / rate_hz
        self.head = head  # latest head pose, updated by the vision loop (None = do not move the head)
        self.body_yaw_deg = body_yaw_deg  # body rotation in degrees (None = leave the body where it is)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="target-sender", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            body = self.body_yaw_deg
            self._mini.set_target(head=self.head, antennas=antennas_with_neutral(self._reaction.angles(now)),
                                  body_yaw=None if body is None else math.radians(body))
            self._stop.wait(max(0.0, self._interval - (time.perf_counter() - now)))
