"""Interactive and expressive head and antenna gestures for workout session phases."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

from core.motion.antennas import NEUTRAL_ANTENNAS
from core.motion.voice import ActionVoice


@dataclass
class GestureStep:
    """One keyframe step in an expressive robot gesture."""

    head_kwargs: dict | None  # kwargs for create_head_pose(pitch=..., roll=..., yaw=...)
    antennas: list[float]     # [right, left] in radians
    duration: float           # transition duration in seconds


# --- Gestures definition ---------------------------------------------------

WELCOME_GESTURE: list[GestureStep] = [
    # Friendly head lift and inquisitive tilt with antennas perking up
    GestureStep({"pitch": -8, "roll": 6, "degrees": True}, [0.6, 0.6], 0.6),
    GestureStep({"pitch": -4, "roll": -6, "degrees": True}, [0.4, -0.4], 0.6),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.5),
]

ROUND_INTRO_GESTURE: list[GestureStep] = [
    # Attentive, ready forward nod with alert antennas
    GestureStep({"pitch": 10, "roll": 0, "yaw": 0, "degrees": True}, [0.5, 0.5], 0.5),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
]

REP_BEAT_GESTURE: list[GestureStep] = [
    # Dynamic nod + antenna pop celebrating a detected repetition
    GestureStep({"pitch": 12, "roll": 0, "degrees": True}, [0.8, -0.8], 0.25),
    GestureStep({"pitch": -4, "roll": 0, "degrees": True}, [-0.6, 0.6], 0.25),
    GestureStep({"pitch": 0, "roll": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.3),
]

ROUND_COMPLETE_GESTURE: list[GestureStep] = [
    # Cheerful side-to-side tilt with celebration wave
    GestureStep({"pitch": -6, "roll": 15, "degrees": True}, [0.7, -0.7], 0.45),
    GestureStep({"pitch": -6, "roll": -15, "degrees": True}, [-0.7, 0.7], 0.45),
    GestureStep({"pitch": 10, "roll": 0, "degrees": True}, [0.6, 0.6], 0.45),
    GestureStep({"pitch": 0, "roll": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
]

OUTRO_CELEBRATION_GESTURE: list[GestureStep] = [
    # Grand finale celebration: looking up triumphantly, waving antennas, celebratory dance, and bow
    GestureStep({"pitch": -15, "roll": 0, "yaw": 0, "degrees": True}, [0.8, 0.8], 0.65),
    GestureStep({"pitch": -8, "roll": 18, "yaw": 12, "degrees": True}, [0.8, -0.8], 0.5),
    GestureStep({"pitch": -8, "roll": -18, "yaw": -12, "degrees": True}, [-0.8, 0.8], 0.5),
    GestureStep({"pitch": 15, "roll": 0, "yaw": 0, "z": -15, "mm": True, "degrees": True}, [-0.3, -0.3], 0.7),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.8),
]

GESTURES: dict[str, list[GestureStep]] = {
    "welcome": WELCOME_GESTURE,
    "round_intro": ROUND_INTRO_GESTURE,
    "rep_beat": REP_BEAT_GESTURE,
    "round_complete": ROUND_COMPLETE_GESTURE,
    "outro": OUTRO_CELEBRATION_GESTURE,
}


def play_gesture_with_speech(
    mini,
    gesture: Sequence[GestureStep],
    voice: ActionVoice | None = None,
    speech_text: str | None = None,
    wait_for_voice: bool = True,
) -> None:
    """Execute a coordinated expressive gesture and speech concurrently.

    The speech starts asynchronously via `voice.speak()` so robot movement and
    voice output happen simultaneously in an embodied, lively manner.

    Args:
        mini: ReachyMini robot instance (simulation or physical).
        gesture: Sequence of `GestureStep` keyframes.
        voice: Optional ActionVoice announcer.
        speech_text: Optional sentence to speak alongside the gesture.
        wait_for_voice: Whether to wait until speech playback completes before returning.
    """
    from reachy_mini.utils import create_head_pose

    # Start speech immediately in the background
    if voice is not None and speech_text:
        voice.speak(speech_text, non_blocking=True)

    # Execute motion keyframes
    if mini is not None:
        for step in gesture:
            head_target = (
                create_head_pose(**step.head_kwargs)
                if step.head_kwargs is not None
                else create_head_pose()
            )
            mini.goto_target(
                head=head_target,
                antennas=step.antennas,
                duration=step.duration,
            )

    # Ensure speech completes gracefully
    if wait_for_voice and voice is not None and speech_text:
        voice.wait(timeout=12.0)
