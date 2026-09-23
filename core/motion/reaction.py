"""Robot reaction to a detected wave: the antennas wave back (non-blocking, sampled by the control loop)."""

import math

# Antenna pose of a woken-up robot ([right, left] in radians); the wave is added around it.
NEUTRAL_ANTENNAS = (-0.1745, 0.1745)


def antennas_with_neutral(wave_angles, neutral=NEUTRAL_ANTENNAS):
    """[right, left] targets: the wave offsets added to the neutral pose (neutral itself when not waving)."""
    if wave_angles is None:
        return list(neutral)
    return [neutral[0] + wave_angles[0], neutral[1] + wave_angles[1]]


class AntennaWave:
    # cooldown_s is dead time *after* the wave ends: keep it short so a second wave is answered quickly
    # (the detector itself also needs ~1 s for the 1.5 s window to stop matching the previous wave).
    def __init__(self, amplitude_deg=30.0, freq_hz=2.5, duration_s=1.2, cooldown_s=0.5):
        self.amplitude_deg = amplitude_deg
        self.freq_hz = freq_hz
        self.duration_s = duration_s
        self.cooldown_s = cooldown_s
        self._start = None

    def trigger(self, now):
        """Start a wave unless one is running or the cooldown after the last one has not passed."""
        if self._start is not None and now < self._start + self.duration_s + self.cooldown_s:
            return False
        self._start = now
        return True

    def active(self, now):
        return self._start is not None and now - self._start < self.duration_s

    def angles(self, now):
        """[right, left] antenna angles in radians (opposite phase), or None when not waving."""
        if not self.active(now):
            return None
        angle = math.radians(self.amplitude_deg) * math.sin(2 * math.pi * self.freq_hz * (now - self._start))
        return [angle, -angle]
