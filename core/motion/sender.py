"""Sends targets to the robot at a steady rate, independent of the slow vision loop."""

import math
import threading
import time


class TargetSender:
    """Context manager: while open, `mini.set_target` is called `rate_hz` times per second.

    Attributes:
        head: Latest head pose set by the vision loop (None = do not move the head).
        body_yaw_deg: Body rotation in degrees (None = leave the body where it is).
    """

    def __init__(self, mini, antennas, rate_hz=50.0, head=None, body_yaw_deg=None):
        """Prepare the sender thread.

        Args:
            mini: The `ReachyMini` connection.
            antennas: Callable `now -> [right, left]` giving the antenna target in radians.
            rate_hz: Sends per second (the daemon runs at 50 Hz).
            head: Initial head pose.
            body_yaw_deg: Initial body yaw.
        """
        self._mini = mini
        self._antennas = antennas
        self._interval = 1.0 / rate_hz
        self.head = head
        self.body_yaw_deg = body_yaw_deg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="target-sender", daemon=True)

    def start(self):
        """Start the thread and return self."""
        self._thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def stop(self):
        """Stop the thread and wait for it."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            body = self.body_yaw_deg
            self._mini.set_target(head=self.head, antennas=self._antennas(now),
                                  body_yaw=None if body is None else math.radians(body))
            self._stop.wait(max(0.0, self._interval - (time.perf_counter() - now)))
