"""Session plan configuration, exercise definitions, and dynamic singular/plural phrasing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

DEFAULT_REPS_PER_ROUND: int = 1
DEFAULT_TIMEOUT_S: float = 30.0

# CMAS items 7/8: a held posture timed by `core.motion.cmas.OverheadHold`, not a classifier class.
HOLD_ACTION: str = "arms_up"
HOLD_TIMEOUT_S: float = 60.0
HOLD_MIN_S: float = 1.0  # a shorter hold is a false start: the round keeps waiting
HOLD_CHEER_EVERY_S: int = 5

# Supported demonstrable exercises in the repository
SUPPORTED_EXERCISES: tuple[str, ...] = ("squat", "clapping", "jumping_jacks", "wave")
CMAS_EXERCISES: tuple[str, ...] = ("chair_rise", "pick_up", HOLD_ACTION)

EXERCISE_NOUNS: dict[str, tuple[str, str]] = {
    # action_key: (singular, plural)
    "squat":         ("squat", "squats"),
    "clapping":      ("clap", "claps"),
    "jumping_jacks": ("jumping jack", "jumping jacks"),
    "wave":          ("wave", "waves"),
    "pushup":        ("push-up", "push-ups"),
    "arm_raise":     ("arm raise", "arm raises"),
    "side_bend":     ("side bend", "side bends"),
    "chair_rise":    ("chair rise", "chair rises"),
    "pick_up":       ("pick-up", "pick-ups"),
    HOLD_ACTION:     ("arms up", "arms up"),
}


def get_nouns(action: str) -> tuple[str, str]:
    """Return (singular, plural) nouns for an action name."""
    return EXERCISE_NOUNS.get(action, (action, f"{action}s"))


def format_exercise_name(action: str, count: int) -> str:
    """Format action name matching the given count (singular vs plural)."""
    sing, plur = get_nouns(action)
    noun = sing if count == 1 else plur
    return f"{count} {noun}"


def get_welcome_speech(num_rounds: int, reps_per_round: int) -> str:
    """Generate friendly welcome speech with grammatical singular/plural agreement."""
    round_phrase = "1 round" if num_rounds == 1 else f"{num_rounds} rounds"
    if reps_per_round == 1:
        rep_phrase = "1 repetition each" if num_rounds > 1 else "1 repetition"
    else:
        rep_phrase = f"{reps_per_round} repetitions each" if num_rounds > 1 else f"{reps_per_round} repetitions"

    return f"Let's play a game! We'll do {round_phrase}, with {rep_phrase}. Ready? Let's go!"


def get_round_intro_speech(round_idx: int, total_rounds: int, action: str, target_reps: int) -> str:
    """Generate round introduction speech with singular/plural agreement."""
    target_phrase = format_exercise_name(action, target_reps)
    if total_rounds > 1:
        return f"Round {round_idx} of {total_rounds}! Show me {target_phrase}!"
    return f"Show me {target_phrase}!"


def get_round_announcement_speech(round_idx: int, total_rounds: int, action: str) -> str:
    """Announce the round before the robot demonstrates the exercise."""
    singular, plural = get_nouns(action)
    exercise_name = singular if singular == plural else plural
    if total_rounds > 1:
        return f"Round {round_idx} of {total_rounds}. Next up: {exercise_name}."
    return f"Next up: {exercise_name}."


def get_exercise_demo_speech(action: str) -> str:
    """Tell the user the robot will demonstrate the movement before showing it."""
    prompts = {
        "squat": "Before you try, I'll show you how squats work. Watch my head move down, then up!",
        "arm_raise": "Before you try, I'll show you an arm raise. Watch my antennas move out to the sides, then return upright!",
        "side_bend": "Before you try, I'll show you a side bend. Watch my head and body move from side to side!",
        "clapping": "Before you try, I'll show you clapping. Watch my antennas move together and open again!",
        "jumping_jacks": "Before you try, I'll show you jumping jacks. Watch my antennas open out and return!",
        "wave": "Before you try, I'll show you a wave. Watch me wave my antenna!",
        "pushup": "Before you try, I'll show you a push-up. Watch my head dip down and rise again!",
        "push_up": "Before you try, I'll show you a push-up. Watch my head dip down and rise again!",
    }
    normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
    return prompts.get(normalized, "Before you try, I'll show you how this exercise works. Watch me!")


def get_round_prompt_speech(action: str, target_reps: int) -> str:
    """Prompt the user to begin after the robot has finished its demonstration."""
    if action == HOLD_ACTION:
        return "Raise both arms straight above your head and hold them up as long as you can!"
    if action == "chair_rise":
        return f"Sit on the chair, then stand up slowly. Show me {format_exercise_name(action, target_reps)}!"
    if action == "pick_up":
        return f"Pick something up from the floor. Show me {format_exercise_name(action, target_reps)}!"
    return f"Now it's your turn! Show me {format_exercise_name(action, target_reps)}!"


def get_rep_speech(rep_num: int, target_reps: int) -> str:
    """Generate real-time repetition counting feedback."""
    number_words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
    }
    word = number_words.get(rep_num, str(rep_num))

    if target_reps == 1:
        return f"{word}! You did it!"
    elif rep_num >= target_reps:
        return f"{word}! You did it!"
    else:
        return f"{word}!"


def get_round_complete_speech(action: str, reps_done: int, timed_out: bool = False) -> str:
    """Generate round feedback upon completion or timeout."""
    if timed_out and reps_done == 0:
        return "Time's up! That's okay, let's try the next one!"

    done_phrase = format_exercise_name(action, reps_done)
    if timed_out:
        return f"Time's up! You did {done_phrase}. Good try!"
    return f"Yay! You did {done_phrase}!"


def get_hold_cheer_speech(seconds: int) -> str:
    """Encourage the user while the arms are up; `seconds` is a multiple of HOLD_CHEER_EVERY_S."""
    cheers = ("Keep going!", "You're so strong!", "Amazing!", "Don't stop now!")
    return f"{seconds} seconds! {cheers[(seconds // HOLD_CHEER_EVERY_S - 1) % len(cheers)]}"


def get_hold_complete_speech(held_s: float) -> str:
    """Feedback after the arms came down (or the round ran out), from the longest hold."""
    seconds = round(held_s)
    if held_s < HOLD_MIN_S:
        return "Time's up! That's okay, let's try the next one!"
    return f"Wow, you kept your arms up for {seconds} {'second' if seconds == 1 else 'seconds'}!"


def get_outro_speech(total_reps: int, completed_rounds: int, total_rounds: int) -> str:
    """Generate celebratory session outro speech with singular/plural agreement."""
    if total_reps == 0:  # e.g. a CMAS session where only the arms-up hold went well
        return "All done! You were amazing! Bye bye, see you next time!"
    rep_word = "repetition" if total_reps == 1 else "repetitions"
    round_word = "round" if completed_rounds == 1 else "rounds"
    return (
        f"All done! You were amazing! You did {total_reps} {rep_word} "
        f"in {completed_rounds} {round_word}. Bye bye, see you next time!"
    )


@dataclass
class ExerciseRound:
    """Configuration for a single exercise round."""

    action: str
    target_reps: int = DEFAULT_REPS_PER_ROUND
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def singular_name(self) -> str:
        return get_nouns(self.action)[0]

    @property
    def plural_name(self) -> str:
        return get_nouns(self.action)[1]


@dataclass
class SessionPlan:
    """Complete exercise session workout plan."""

    rounds: list[ExerciseRound] = field(default_factory=list)
    reps_per_round: int = DEFAULT_REPS_PER_ROUND

    @classmethod
    def create(
        cls,
        exercises: Sequence[str] = SUPPORTED_EXERCISES,
        reps_per_round: int = DEFAULT_REPS_PER_ROUND,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> SessionPlan:
        """Create a session plan with `reps_per_round` repetitions for each exercise."""
        rounds = [
            ExerciseRound(action=ex, target_reps=reps_per_round,
                          timeout_s=HOLD_TIMEOUT_S if ex == HOLD_ACTION else timeout_s)
            for ex in exercises
        ]
        return cls(rounds=rounds, reps_per_round=reps_per_round)
