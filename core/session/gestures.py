"""Interactive and expressive head and antenna gestures for workout session phases."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    body_yaw_deg: float | None = None  # optional body rotation, in degrees


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


# Exercise-specific demonstrations. These are deliberately short and low-amplitude:
# the head and antennas act as visual stand-ins for a person's movement. Body yaw
# is only a horizontal turn (the robot cannot literally bend its torso sideways),
# so the side-bend demo combines a small body turn with a lateral head roll.
ARM_RAISE_HORIZONTAL = [1.35, -1.35]  # [right, left], near-horizontal from the upright pose

SQUAT_DEMO: list[GestureStep] = [
    GestureStep({"pitch": -5, "z": 5, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
    GestureStep({"pitch": 13, "z": -8, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.55),
    GestureStep({"pitch": -7, "z": 6, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.55),
    GestureStep({"pitch": 13, "z": -8, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.55),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "z": 0, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.5),
]

ARM_RAISE_DEMO: list[GestureStep] = [
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.35, 0),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(ARM_RAISE_HORIZONTAL), 0.7, 0),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.7, 0),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(ARM_RAISE_HORIZONTAL), 0.65, 0),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.7, 0),
]

SIDE_BEND_DEMO: list[GestureStep] = [
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.35, 0),
    GestureStep({"pitch": 0, "roll": 15, "yaw": 0, "degrees": True}, [0.25, -0.25], 0.7, 16),
    GestureStep({"pitch": 0, "roll": -15, "yaw": 0, "degrees": True}, [-0.25, 0.25], 0.8, -16),
    GestureStep({"pitch": 0, "roll": 15, "yaw": 0, "degrees": True}, [0.25, -0.25], 0.7, 16),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.55, 0),
]

CLAPPING_DEMO: list[GestureStep] = [
    GestureStep({"pitch": -3, "roll": 0, "yaw": 0, "degrees": True}, [0.55, 0.55], 0.35),
    GestureStep({"pitch": 5, "roll": 0, "yaw": 0, "degrees": True}, [-0.55, -0.55], 0.28),
    GestureStep({"pitch": -3, "roll": 0, "yaw": 0, "degrees": True}, [0.55, 0.55], 0.28),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
]

JUMPING_JACKS_DEMO: list[GestureStep] = [
    GestureStep({"pitch": -5, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.35),
    GestureStep({"pitch": -10, "roll": 0, "yaw": 0, "degrees": True}, list(ARM_RAISE_HORIZONTAL), 0.55),
    GestureStep({"pitch": 7, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.55),
    GestureStep({"pitch": -10, "roll": 0, "yaw": 0, "degrees": True}, list(ARM_RAISE_HORIZONTAL), 0.55),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.5),
]

WAVE_DEMO: list[GestureStep] = [
    GestureStep({"pitch": -5, "roll": 0, "yaw": 12, "degrees": True}, [0.85, NEUTRAL_ANTENNAS[1]], 0.35),
    GestureStep({"pitch": -5, "roll": 0, "yaw": -12, "degrees": True}, [-0.85, NEUTRAL_ANTENNAS[1]], 0.35),
    GestureStep({"pitch": -5, "roll": 0, "yaw": 12, "degrees": True}, [0.85, NEUTRAL_ANTENNAS[1]], 0.35),
    GestureStep({"pitch": 0, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.45),
]

PUSHUP_DEMO: list[GestureStep] = [
    GestureStep({"pitch": -5, "z": 4, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
    GestureStep({"pitch": 14, "z": -8, "mm": True, "degrees": True}, [0.15, -0.15], 0.6),
    GestureStep({"pitch": -5, "z": 4, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.6),
    GestureStep({"pitch": 0, "z": 0, "mm": True, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
]

DEFAULT_EXERCISE_DEMO: list[GestureStep] = [
    GestureStep({"pitch": 10, "roll": 0, "yaw": 0, "degrees": True}, [0.35, -0.35], 0.35),
    GestureStep({"pitch": -4, "roll": 0, "yaw": 0, "degrees": True}, list(NEUTRAL_ANTENNAS), 0.4),
]

EXERCISE_DEMOS: dict[str, list[GestureStep]] = {
    "squat": SQUAT_DEMO,
    "arm_raise": ARM_RAISE_DEMO,
    "side_bend": SIDE_BEND_DEMO,
    "clapping": CLAPPING_DEMO,
    "jumping_jacks": JUMPING_JACKS_DEMO,
    "wave": WAVE_DEMO,
    "pushup": PUSHUP_DEMO,
    "push_up": PUSHUP_DEMO,
}

GESTURES: dict[str, list[GestureStep]] = {
    "welcome": WELCOME_GESTURE,
    "round_intro": ROUND_INTRO_GESTURE,
    "rep_beat": REP_BEAT_GESTURE,
    "round_complete": ROUND_COMPLETE_GESTURE,
    "outro": OUTRO_CELEBRATION_GESTURE,
}


def play_gesture(mini, gesture: Sequence[GestureStep]) -> None:
    """Play a gesture without speech. Optional body_yaw_deg enables body choreography."""
    from reachy_mini.utils import create_head_pose

    if mini is None:
        return

    for step in gesture:
        head_target = (
            create_head_pose(**step.head_kwargs)
            if step.head_kwargs is not None
            else create_head_pose()
        )
        kwargs = {
            "head": head_target,
            "antennas": step.antennas,
            "duration": step.duration,
        }
        if step.body_yaw_deg is not None:
            kwargs["body_yaw"] = math.radians(step.body_yaw_deg)
        mini.goto_target(**kwargs)


def play_exercise_demo(mini, action: str) -> None:
    """Demonstrate an exercise with a matching head, body, and/or antenna movement."""
    normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
    play_gesture(mini, EXERCISE_DEMOS.get(normalized, DEFAULT_EXERCISE_DEMO))


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
    # Start speech immediately in the background
    if voice is not None and speech_text:
        voice.speak(speech_text, non_blocking=True)

    # Execute motion keyframes (same choreography helper also supports body yaw).
    play_gesture(mini, gesture)

    # Ensure speech completes gracefully
    if wait_for_voice and voice is not None and speech_text:
        voice.wait(timeout=12.0)
