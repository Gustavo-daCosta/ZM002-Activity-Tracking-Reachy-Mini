"""Voice announcer for recognised actions and robot speech — natural offline neural TTS.

Provides high-quality, natural neural speech using Piper TTS (ONNX/VITS) by default,
with graceful fallback to spd-say (speech-dispatcher) when Piper is not available.

Features:
- Natural offline neural voice synthesis using Piper TTS (~50 ms synthesis latency).
- Non-blocking execution via background worker queue: vision loops and robot motor
  threads are never blocked or delayed by audio generation or playback.
- Cooldown tracking per action class so repeated detections do not trigger spam.
- General `speak(text)` method for arbitrary announcements (workout greetings, reps).
- Automatic on-demand downloading of Piper voice models (.onnx and .onnx.json).
- Multi-backend fallback (Piper -> spd-say -> print/no-op).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional
import urllib.request

logger = logging.getLogger(__name__)

# Default spoken label per action class (must match core/motion/actions.ACTIONS).
DEFAULT_LABELS: dict[str, str] = {
    "wave":          "Wave detected",
    "pushup":        "Push-up detected",
    "squat":         "Squat detected",
    "clapping":      "Clapping detected",
    "jumping_jacks": "Jumping jacks detected",
}

DEFAULT_RATE: int = -30          # spd-say rate (-100 to 100)
DEFAULT_LANGUAGE: str = "en"
DEFAULT_VOICE: str = "en_US-lessac-medium"

# Storage directory for downloaded Piper voice models
VOICES_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "voices"

# Well-known Piper voices hosted on Hugging Face (rhasspy/piper-voices)
HF_PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

KNOWN_VOICE_PATHS: dict[str, str] = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
    "en_US-lessac-low":    "en/en_US/lessac/low/en_US-lessac-low",
    "en_US-amy-medium":       "en/en_US/amy/medium/en_US-amy-medium",
    "en_US-amy-low":          "en/en_US/amy/low/en_US-amy-low",
    "en_GB-alba-medium":      "en/en_GB/alba/medium/en_GB-alba-medium",
    "pt_BR-faber-medium":     "pt/pt_BR/faber/medium/pt_BR-faber-medium",
    "pt_BR-edresson-low":     "pt/pt_BR/edresson/low/pt_BR-edresson-low",
    "pt_PT-tuur-medium":      "pt/pt_PT/tuur/medium/pt_PT-tuur-medium",
}


def resolve_voice_urls(voice_name: str) -> tuple[str, str]:
    """Return (onnx_url, json_url) for a given Piper voice name."""
    rel_path = KNOWN_VOICE_PATHS.get(voice_name)
    if rel_path is None:
        # Infer standard structure: {lang}_{REGION}-{speaker}-{quality}
        # Example: es_ES-carlfm-x_low -> es/es_ES/carlfm/x_low/es_ES-carlfm-x_low
        parts = voice_name.split("-")
        if len(parts) >= 3:
            locale = parts[0]
            speaker = parts[1]
            quality = "-".join(parts[2:])
            lang = locale.split("_")[0]
            rel_path = f"{lang}/{locale}/{speaker}/{quality}/{voice_name}"
        else:
            rel_path = f"{voice_name}/{voice_name}"

    onnx_url = f"{HF_PIPER_BASE_URL}/{rel_path}.onnx"
    json_url = f"{HF_PIPER_BASE_URL}/{rel_path}.onnx.json"
    return onnx_url, json_url


def ensure_voice_model(voice_name: str = DEFAULT_VOICE, target_dir: Path | None = None) -> tuple[Path, Path]:
    """Ensure that the requested Piper model and config exist locally, downloading if needed.

    Returns:
        tuple (model_path, config_path)
    """
    dest_dir = target_dir or VOICES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_path = dest_dir / f"{voice_name}.onnx"
    config_path = dest_dir / f"{voice_name}.onnx.json"

    if model_path.is_file() and config_path.is_file():
        return model_path, config_path

    onnx_url, json_url = resolve_voice_urls(voice_name)
    headers = {"User-Agent": "Mozilla/5.0 (ReachyMini; TTS)"}

    for url, path in [(onnx_url, model_path), (json_url, config_path)]:
        if path.is_file():
            continue
        temp_path = path.with_name(path.name + ".part")
        logger.info("Downloading Piper voice file: %s -> %s", url, path)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            temp_path.rename(path)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download Piper voice file from {url}: {exc}"
            ) from exc

    return model_path, config_path


class ActionVoice:
    """Speaks action labels and arbitrary text using natural neural TTS or system fallback.

    Architecture:
    - Worker thread + FIFO queue: all speech synthesis and audio output happen
      asynchronously without delaying the main vision or robot control loops.
    - Cooldown management: independent per action class to prevent repetitive announcements.
    - Robust audio backend: prefers Piper TTS + sounddevice/aplay; falls back to spd-say.
    """

    def __init__(
        self,
        labels: dict[str, str] | None = None,
        rate: int = DEFAULT_RATE,
        language: str = DEFAULT_LANGUAGE,
        cooldown_s: float = 3.0,
        voice_model: str = DEFAULT_VOICE,
        backend: str = "auto",
        length_scale: float = 1.0,
        volume: float = 1.0,
    ) -> None:
        """Initialize the voice announcer.

        Args:
            labels: Mapping action name -> text to speak. Defaults to DEFAULT_LABELS.
            rate: spd-say rate (-100..+100) if fallback backend is used.
            language: Language code passed to spd-say.
            cooldown_s: Minimum gap between two announcements of the same action.
            voice_model: Piper model name (e.g. "en_US-lessac-medium").
            backend: "auto" (prefers piper, falls back to spd-say), "piper", or "spd-say".
            length_scale: Piper speech rate multiplier (1.0 = normal, 0.9 = faster, 1.1 = slower).
            volume: Audio volume (1.0 = 100%).
        """
        self.labels: dict[str, str] = dict(DEFAULT_LABELS if labels is None else labels)
        self.rate = rate
        self.language = language
        self.cooldown_s = cooldown_s
        self.voice_model_name = voice_model
        self.length_scale = length_scale
        self.volume = volume

        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()

        # Backend state
        self.backend: str = "none"
        self.available: bool = False
        self._piper_voice = None
        self._syn_config = None

        # Queue for sequential speech synthesis & playback
        self._queue: queue.Queue[tuple[str, Optional[threading.Event]]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._init_backend(backend)

        if self.available:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="ActionVoice-Worker"
            )
            self._worker_thread.start()

    def _init_backend(self, requested_backend: str) -> None:
        """Initialize the selected TTS backend."""
        # 1. Attempt Piper TTS if requested or auto
        if requested_backend in ("auto", "piper"):
            try:
                import piper
                from piper.config import SynthesisConfig

                model_path, config_path = ensure_voice_model(self.voice_model_name)
                self._piper_voice = piper.PiperVoice.load(model_path, config_path)
                self._syn_config = SynthesisConfig(
                    length_scale=self.length_scale,
                    volume=self.volume,
                )
                self.backend = "piper"
                self.available = True
                return
            except Exception as exc:
                logger.warning("Could not initialize Piper TTS (%s): %s", self.voice_model_name, exc)
                if requested_backend == "piper":
                    self.available = False
                    return

        # 2. Attempt spd-say fallback
        if requested_backend in ("auto", "spd-say"):
            if shutil.which("spd-say") is not None:
                self.backend = "spd-say"
                self.available = True
                return

        self.backend = "none"
        self.available = False

    def trigger(self, action: str, now: float) -> bool:
        """Speak the label for *action* unless in cooldown.

        Returns True when an announcement was queued; False otherwise.
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

        return self.speak(label, non_blocking=True)

    def speak(self, text: str, non_blocking: bool = True) -> bool:
        """Speak arbitrary text aloud.

        Args:
            text: Text sentence to synthesize and speak.
            non_blocking: True to queue and return immediately; False to wait until playback finishes.

        Returns:
            True if text was queued/spoken; False if TTS is unavailable.
        """
        if not self.available or not text.strip():
            return False

        done_event = threading.Event() if not non_blocking else None
        self._queue.put((text, done_event))

        if not non_blocking and done_event is not None:
            done_event.wait()

        return True

    # Sentinel value used by wait() to signal "drain up to here"
    _FENCE = "__voice_fence__"

    def wait(self, timeout: float | None = 15.0) -> None:
        """Wait until all currently queued speech has completed, with timeout.

        Uses a fence sentinel so we never block forever if the audio backend
        hangs.  The default timeout of 15 s is generous enough for any sentence.
        """
        if not self.available:
            return
        done = threading.Event()
        self._queue.put((self._FENCE, done))
        if not done.wait(timeout=timeout):
            logger.warning("voice.wait() timed out after %.1f s — audio backend may be stuck", timeout)

    def _worker_loop(self) -> None:
        """Background thread consuming text items and playing audio sequentially."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            text, done_event = item
            try:
                if text == self._FENCE:
                    # Fence sentinel: just signal, don't speak anything.
                    pass
                else:
                    self._speak_sync(text)
            except Exception as exc:
                logger.error("Error during TTS playback: %s", exc)
            finally:
                self._queue.task_done()
                if done_event is not None:
                    done_event.set()

    def _speak_sync(self, text: str) -> None:
        """Synchronously synthesize and play text."""
        if self.backend == "piper" and self._piper_voice is not None:
            self._speak_piper(text)
        elif self.backend == "spd-say":
            self._speak_spd_say(text)

    def _speak_piper(self, text: str) -> None:
        """Synthesize with Piper and output audio, with hang protection."""
        import numpy as np

        chunks = list(self._piper_voice.synthesize(text, syn_config=self._syn_config))
        if not chunks:
            return

        audio = np.concatenate([c.audio_float_array for c in chunks])
        sample_rate = chunks[0].sample_rate
        audio_duration_s = len(audio) / sample_rate
        # Give 5 extra seconds beyond the natural audio length before we give up.
        playback_timeout = audio_duration_s + 5.0

        # Try sounddevice first, guarded by a thread so sd.wait() can't hang forever.
        try:
            import sounddevice as sd

            played = threading.Event()

            def _play() -> None:
                try:
                    sd.play(audio, sample_rate)
                    sd.wait()
                finally:
                    played.set()

            t = threading.Thread(target=_play, daemon=True, name="sd-playback")
            t.start()
            if played.wait(timeout=playback_timeout):
                return  # finished successfully
            # Timed out — attempt a graceful stop and fall through to subprocess.
            logger.warning("sounddevice playback timed out (%.1f s); falling back", playback_timeout)
            try:
                sd.stop()
            except Exception:
                pass
        except Exception as sd_exc:
            logger.debug("sounddevice unavailable (%s), falling back to system player", sd_exc)

        # Fallback to system audio players (aplay / pw-play).
        int16_bytes = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if shutil.which("pw-play") is not None:
            try:
                subprocess.run(
                    ["pw-play", "--rate", str(sample_rate), "--channels", "1", "--format", "s16"],
                    input=int16_bytes,
                    check=False,
                    timeout=playback_timeout,
                )
            except subprocess.TimeoutExpired:
                logger.warning("pw-play timed out")
        elif shutil.which("aplay") is not None:
            try:
                subprocess.run(
                    ["aplay", "-q", "-r", str(sample_rate), "-f", "S16_LE", "-c", "1"],
                    input=int16_bytes,
                    check=False,
                    timeout=playback_timeout,
                )
            except subprocess.TimeoutExpired:
                logger.warning("aplay timed out")

    def _speak_spd_say(self, text: str) -> None:
        """Run spd-say synchronously."""
        subprocess.run(
            ["spd-say", "--rate", str(self.rate), "--language", self.language, "--", text],
            check=False,
            timeout=30,
        )

    def close(self) -> None:
        """Stop worker thread and cleanup."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
