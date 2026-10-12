"""Interactive and expressive head and antenna gestures for workout session phases."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from core.motion.antennas import NEUTRAL_ANTENNAS
from core.motion.voice import ActionVoice


@dataclass
class GestureStep:
    """One keyframe step in an expressive robot gesture.

    Components are optional so a gesture can move only the head,
    only the antennas, or only the body.
    """

    head_kwargs: dict | None
    antennas: list[float] | None  # [right, left], in radians
    duration: float
    body_yaw_deg: float | None = None


# ---------------------------------------------------------------------------
# General gestures
# ---------------------------------------------------------------------------

WELCOME_GESTURE: list[GestureStep] = [
    GestureStep(
        {"pitch": -8, "roll": 6, "degrees": True},
        [0.6, 0.6],
        0.6,
    ),
    GestureStep(
        {"pitch": -4, "roll": -6, "degrees": True},
        [0.4, -0.4],
        0.6,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.5,
    ),
]

ROUND_INTRO_GESTURE: list[GestureStep] = [
    GestureStep(
        {"pitch": 10, "roll": 0, "yaw": 0, "degrees": True},
        [0.5, 0.5],
        0.5,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.4,
    ),
]

REP_BEAT_GESTURE: list[GestureStep] = [
    GestureStep(
        {"pitch": 12, "roll": 0, "degrees": True},
        [0.8, -0.8],
        0.25,
    ),
    GestureStep(
        {"pitch": -4, "roll": 0, "degrees": True},
        [-0.6, 0.6],
        0.25,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.3,
    ),
]

ROUND_COMPLETE_GESTURE: list[GestureStep] = [
    GestureStep(
        {"pitch": -6, "roll": 15, "degrees": True},
        [0.7, -0.7],
        0.45,
    ),
    GestureStep(
        {"pitch": -6, "roll": -15, "degrees": True},
        [-0.7, 0.7],
        0.45,
    ),
    GestureStep(
        {"pitch": 10, "roll": 0, "degrees": True},
        [0.6, 0.6],
        0.45,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.4,
    ),
]

OUTRO_CELEBRATION_GESTURE: list[GestureStep] = [
    GestureStep(
        {"pitch": -15, "roll": 0, "yaw": 0, "degrees": True},
        [0.8, 0.8],
        0.65,
    ),
    GestureStep(
        {"pitch": -8, "roll": 18, "yaw": 12, "degrees": True},
        [0.8, -0.8],
        0.5,
    ),
    GestureStep(
        {"pitch": -8, "roll": -18, "yaw": -12, "degrees": True},
        [-0.8, 0.8],
        0.5,
    ),
    GestureStep(
        {
            "pitch": 15,
            "roll": 0,
            "yaw": 0,
            "z": -15,
            "mm": True,
            "degrees": True,
        },
        [-0.3, -0.3],
        0.7,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.8,
    ),
]


# ---------------------------------------------------------------------------
# Exercise demonstrations
# ---------------------------------------------------------------------------

# Antenna values are [right antenna, left antenna].
#
# The opposing signs make the antennae move outward in opposite directions.
# 1.35 radians is approximately 77 degrees, a near-horizontal target that
# stays slightly short of a full 90-degree movement.
ARM_RAISE_HORIZONTAL = [1.35, -1.35]


# Squat:
# Move the head vertically down and return it to its neutral height.
# There is no intentional head tilt, antenna motion, or body rotation.
SQUAT_DEMO: list[GestureStep] = [
    GestureStep(
        {
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "z": -10,
            "mm": True,
            "degrees": True,
        },
        None,
        0.6,
    ),
    GestureStep(
        {
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "z": 0,
            "mm": True,
            "degrees": True,
        },
        None,
        0.6,
    ),
    GestureStep(
        {
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "z": -10,
            "mm": True,
            "degrees": True,
        },
        None,
        0.6,
    ),
    GestureStep(
        {
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "z": 0,
            "mm": True,
            "degrees": True,
        },
        None,
        0.6,
    ),
]


# Antenna joint positions are [left, right], in radians.
#
# Sleep position:
#   Both antennas hang down close to the robot's body.
#
# Arm raise:
#   Both antennas rotate outward by approximately 90 degrees,
#   so they extend horizontally away from the body.

ARM_RAISE_SLEEP = [-3.05, 3.05]

ARM_RAISE_HORIZONTAL = [
    -math.pi / 2,
    math.pi / 2,
]


ARM_RAISE_DEMO: list[GestureStep] = [
    # 1. Start with both antennas hanging down in the sleep position.
    GestureStep(
        head_kwargs=None,
        antennas=list(ARM_RAISE_SLEEP),
        duration=0.8,
    ),

    # 2. Raise both antennas outward to the horizontal position.
    GestureStep(
        head_kwargs=None,
        antennas=list(ARM_RAISE_HORIZONTAL),
        duration=1.0,
    ),

    # 3. Lower both antennas back to the hanging sleep position.
    GestureStep(
        head_kwargs=None,
        antennas=list(ARM_RAISE_SLEEP),
        duration=1.0,
    ),

    # 4. Demonstrate the arm raise a second time.
    GestureStep(
        head_kwargs=None,
        antennas=list(ARM_RAISE_HORIZONTAL),
        duration=1.0,
    ),

    # 5. Finish with both antennas hanging down.
    GestureStep(
        head_kwargs=None,
        antennas=list(ARM_RAISE_SLEEP),
        duration=1.0,
    ),
]


# Side bend:
# Tilt the head left and right using roll only.
# Antennae and body rotation are not commanded.
SIDE_BEND_DEMO: list[GestureStep] = [
    GestureStep(
        {
            "pitch": 0,
            "roll": 18,
            "yaw": 0,
            "degrees": True,
        },
        None,
        0.6,
    ),
    GestureStep(
        {
            "pitch": 0,
            "roll": -18,
            "yaw": 0,
            "degrees": True,
        },
        None,
        0.8,
    ),
    GestureStep(
        {
            "pitch": 0,
            "roll": 18,
            "yaw": 0,
            "degrees": True,
        },
        None,
        0.8,
    ),
    GestureStep(
        {
            "pitch": 0,
            "roll": 0,
            "yaw": 0,
            "degrees": True,
        },
        None,
        0.6,
    ),
]


# Clapping: a simple antenna movement accompanied by a small head nod.
CLAPPING_DEMO: list[GestureStep] = [
    GestureStep(
        {"pitch": -3, "roll": 0, "yaw": 0, "degrees": True},
        [0.55, 0.55],
        0.35,
    ),
    GestureStep(
        {"pitch": 5, "roll": 0, "yaw": 0, "degrees": True},
        [-0.55, -0.55],
        0.28,
    ),
    GestureStep(
        {"pitch": -3, "roll": 0, "yaw": 0, "degrees": True},
        [0.55, 0.55],
        0.28,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.4,
    ),
]


# Jumping jacks: antennae extend and return while the head bobs.
JUMPING_JACKS_DEMO: list[GestureStep] = [
    GestureStep(
        {"pitch": -5, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.35,
    ),
    GestureStep(
        {"pitch": -10, "roll": 0, "yaw": 0, "degrees": True},
        list(ARM_RAISE_HORIZONTAL),
        0.55,
    ),
    GestureStep(
        {"pitch": 7, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.55,
    ),
    GestureStep(
        {"pitch": -10, "roll": 0, "yaw": 0, "degrees": True},
        list(ARM_RAISE_HORIZONTAL),
        0.55,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.5,
    ),
]


# Wave: move one antenna while turning the head.
WAVE_DEMO: list[GestureStep] = [
    GestureStep(
        {"pitch": -5, "roll": 0, "yaw": 12, "degrees": True},
        [0.85, NEUTRAL_ANTENNAS[1]],
        0.35,
    ),
    GestureStep(
        {"pitch": -5, "roll": 0, "yaw": -12, "degrees": True},
        [-0.85, NEUTRAL_ANTENNAS[1]],
        0.35,
    ),
    GestureStep(
        {"pitch": -5, "roll": 0, "yaw": 12, "degrees": True},
        [0.85, NEUTRAL_ANTENNAS[1]],
        0.35,
    ),
    GestureStep(
        {"pitch": 0, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.45,
    ),
]


# Push-up: move the head down and up with a small antenna accent.
PUSHUP_DEMO: list[GestureStep] = [
    GestureStep(
        {
            "pitch": -5,
            "z": 4,
            "mm": True,
            "degrees": True,
        },
        list(NEUTRAL_ANTENNAS),
        0.4,
    ),
    GestureStep(
        {
            "pitch": 14,
            "z": -8,
            "mm": True,
            "degrees": True,
        },
        [0.15, -0.15],
        0.6,
    ),
    GestureStep(
        {
            "pitch": -5,
            "z": 4,
            "mm": True,
            "degrees": True,
        },
        list(NEUTRAL_ANTENNAS),
        0.6,
    ),
    GestureStep(
        {
            "pitch": 0,
            "z": 0,
            "mm": True,
            "degrees": True,
        },
        list(NEUTRAL_ANTENNAS),
        0.4,
    ),
]


DEFAULT_EXERCISE_DEMO: list[GestureStep] = [
    GestureStep(
        {"pitch": 10, "roll": 0, "yaw": 0, "degrees": True},
        [0.35, -0.35],
        0.35,
    ),
    GestureStep(
        {"pitch": -4, "roll": 0, "yaw": 0, "degrees": True},
        list(NEUTRAL_ANTENNAS),
        0.4,
    ),
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


# ---------------------------------------------------------------------------
# Gesture execution
# ---------------------------------------------------------------------------

def play_gesture(mini, gesture: Sequence[GestureStep]) -> None:
    """Play a gesture, commanding only the components specified by each step."""
    from reachy_mini.utils import create_head_pose

    if mini is None:
        return

    for step in gesture:
        kwargs = {
            "duration": step.duration,
        }

        # Only command the head when the step specifies a head pose.
        if step.head_kwargs is not None:
            kwargs["head"] = create_head_pose(**step.head_kwargs)

        # Only command the antennae when the step specifies antenna targets.
        if step.antennas is not None:
            kwargs["antennas"] = step.antennas

        # Body yaw is optional and only used in gestures that request it.
        if step.body_yaw_deg is not None:
            kwargs["body_yaw"] = math.radians(step.body_yaw_deg)

        mini.goto_target(**kwargs)


def play_exercise_demo(mini, action: str) -> None:
    """Demonstrate an exercise using its corresponding robot movement."""
    normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
    gesture = EXERCISE_DEMOS.get(normalized, DEFAULT_EXERCISE_DEMO)
    play_gesture(mini, gesture)


def play_gesture_with_speech(
    mini,
    gesture: Sequence[GestureStep],
    voice: ActionVoice | None = None,
    speech_text: str | None = None,
    wait_for_voice: bool = True,
) -> None:
    """Execute a coordinated expressive gesture and speech concurrently.

    Speech begins asynchronously so the robot can move while speaking.

    Args:
        mini: ReachyMini robot instance (simulation or physical).
        gesture: Sequence of GestureStep keyframes.
        voice: Optional ActionVoice announcer.
        speech_text: Optional sentence to speak alongside the gesture.
        wait_for_voice: Whether to wait for speech playback to complete.
    """
    if voice is not None and speech_text:
        voice.speak(speech_text, non_blocking=True)

    play_gesture(mini, gesture)

    if wait_for_voice and voice is not None and speech_text:
        voice.wait(timeout=12.0)