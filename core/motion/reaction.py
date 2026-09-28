"""Robot reaction to a detected wave: the antennas wave back (non-blocking, sampled by the control loop)."""

import math

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
