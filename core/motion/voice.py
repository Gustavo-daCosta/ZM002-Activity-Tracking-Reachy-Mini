"""Voice announcer for recognised actions — the TTS counterpart of AntennaWave.

Mirrors the structure of antennas.py: a self-contained class that manages
cooldown and a background speaking thread so callers never block on audio.

Uses spd-say (speech-dispatcher) when available; silently does nothing
otherwise — robot apps continue to work without speech-dispatcher installed.
This module imports only stdlib, so it ships safely to the Raspberry Pi CM4.
"""

from __future__ import annotations

import shutil
import subprocess
import threading

# Default spoken label per action class (must match core/motion/actions.ACTIONS).
DEFAULT_LABELS: dict[str, str] = {
    "wave":          "Wave detected",
    "pushup":        "Push-up detected",
    "squat":         "Squat detected",
    "clapping":      "Clapping detected",
    "jumping_jacks": "Jumping jacks detected",
}

DEFAULT_RATE: int = -30      # speech rate can go from -100 to 100
DEFAULT_LANGUAGE: str = "en"


class ActionVoice:
    """Speaks the label for a recognised action in a background thread.

    Mirrors AntennaWave.trigger() / AntennaWave.active():

    * ``trigger(action, now)`` — start speaking if not in cooldown.
    * ``available``           — False when spd-say is not on PATH; all
      calls become no-ops, so robot apps work without speech-dispatcher.

    Each action has its own independent cooldown so that two different
    actions can both fire close together.

    Attributes:
        available: True when spd-say is found on PATH at construction time.
        labels: Mapping of action name → text to speak.
        rate: spd-say speech rate (-100..+100).
        language: BCP-47 language code passed to spd-say.
        cooldown_s: Minimum gap (seconds) between two announcements of the
            same action, independent of the antenna cooldown.
    """

    def __init__(
        self,
        labels: dict[str, str] | None = None,
        rate: int = DEFAULT_RATE,
        language: str = DEFAULT_LANGUAGE,
        cooldown_s: float = 3.0,
    ) -> None:
        """Build the announcer.

        Args:
            labels: Mapping action name → text to speak.  None uses
                DEFAULT_LABELS (one sentence per action class in English).
            rate: spd-say rate; -25 is slightly slower than the default.
            language: spd-say language code (e.g. ``"en"``, ``"pt"``).
            cooldown_s: Minimum gap between two announcements of the same
                action.  3 s by default (the action window is 3 s, so this
                prevents the same event from being announced twice).
        """
        self.labels: dict[str, str] = dict(DEFAULT_LABELS if labels is None else labels)
        self.rate = rate
        self.language = language
        self.cooldown_s = cooldown_s
        self.available: bool = shutil.which("spd-say") is not None
        # last time each action was spoken; float("-inf") = never
        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()

    def trigger(self, action: str, now: float) -> bool:
        """Speak the label for *action* unless in cooldown.  Returns True if triggered.

        Non-blocking: the spd-say subprocess runs in a daemon thread so the
        vision loop is never stalled.

        Args:
            action: One of the recognised action class names (``"wave"``,
                ``"pushup"``, ``"squat"``, ``"clapping"``, ``"jumping_jacks"``).
            now: Current time from ``time.perf_counter()``.

        Returns:
            True when an announcement was started; False when not available,
            the action has no label, or the cooldown has not elapsed.
        """
        if not self.available:
            return False
        label = self.labels.get(action)
        if label is None:
            return False

        with self._lock:
            last = self._last_spoken.get(action, float("-inf"))
            if now - last < self.cooldown_s:
                return False
            self._last_spoken[action] = now

        threading.Thread(
            target=self._speak, args=(label,), daemon=True, name=f"voice-{action}"
        ).start()
        return True

    def _speak(self, text: str) -> None:
        """Run spd-say synchronously inside the daemon thread."""
        subprocess.run(
            ["spd-say", "--rate", str(self.rate), "--language", self.language, "--", text],
            check=False,
        )
