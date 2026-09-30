"""Antenna reactions (a sine wave the control loop samples) and the thread that sends targets at 50 Hz."""

import math
import threading
import time

# Antenna pose of a woken-up robot ([right, left] in radians); waves are added around it.
NEUTRAL_ANTENNAS = (-0.1745, 0.1745)


def antennas_with_neutral(wave_angles, neutral=NEUTRAL_ANTENNAS):
    """[right, left] targets: the wave offsets added to the neutral pose (neutral when not waving)."""
    if wave_angles is None:
        return list(neutral)
    return [neutral[0] + wave_angles[0], neutral[1] + wave_angles[1]]


class AntennaWave:
    """A sine wave on the antennas, in opposite phase, with a cooldown after it ends."""

    def __init__(self, amplitude_deg=30.0, freq_hz=2.5, duration_s=1.2, cooldown_s=0.5):
        """Set the wave shape.

        Args:
            amplitude_deg: Peak angle in degrees.
            freq_hz: Oscillation frequency.
            duration_s: How long the wave lasts.
            cooldown_s: Dead time after the wave before another can start.
        """
        self.amplitude_deg = amplitude_deg
        self.freq_hz = freq_hz
        self.duration_s = duration_s
        self.cooldown_s = cooldown_s
        self._start = None

    def trigger(self, now):
        """Start a wave unless one is running or in cooldown. Returns whether it started."""
        if self._start is not None and now < self._start + self.duration_s + self.cooldown_s:
            return False
        self._start = now
        return True

    def active(self, now):
        """Whether a wave is running at `now`."""
        return self._start is not None and now - self._start < self.duration_s

    def angles(self, now):
        """[right, left] antenna angles in radians, or None when not waving."""
        if not self.active(now):
            return None
        angle = math.radians(self.amplitude_deg) * math.sin(2 * math.pi * self.freq_hz * (now - self._start))
        return [angle, -angle]


class TargetSender:
    """Context manager: while open, `mini.set_target` is called `rate_hz` times per second.

    The vision loop runs at ~9 FPS, far too slow to sample a 1.5 Hz antenna wave smoothly.

    Attributes:
        head: Latest head pose set by the vision loop (None = do not move the head).
        body_yaw_deg: Body rotation in degrees (None = leave the body where it is).
    """

    def __init__(self, mini, antennas, rate_hz=50.0):
        """Prepare the sender thread.

        Args:
            mini: The `ReachyMini` connection.
            antennas: Callable `now -> [right, left]` giving the antenna target in radians.
            rate_hz: Sends per second (the daemon runs at 50 Hz).
        """
        self._mini = mini
        self._antennas = antennas
        self._interval = 1.0 / rate_hz
        self.head = None
        self.body_yaw_deg = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="target-sender", daemon=True)

    def start(self):
        """Start the thread and return self."""
        self._thread.start()
        return self

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
