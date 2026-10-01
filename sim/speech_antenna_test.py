"""Activity recognition announcement test — audio + antennas, no camera.

Simulates what the robot does when it recognises a movement: it speaks the
action name aloud and reacts with a distinct antenna choreography for each
class.  No camera or pose model is involved — this is a pure audio + motor
test that you can run while the MuJoCo simulator is open.

The six classes match the project's trained recogniser
(core/motion/actions.py): none, wave, push-up, squat, clapping, jumping jacks.

Prerequisites:
    # terminal 1 — start the MuJoCo simulator
    sim/start.sh

    # terminal 2 — run this test
    reachy_mini_env/bin/python -m sim.speech_antenna_test

Options:
    --host HOST        daemon host (default: localhost)
    --rate RATE        spd-say speech rate, -100..+100 (default: -25, slower)
    --language LANG    language code for spd-say (default: en)
    --list-actions     print the action list and exit

The script falls back to printing the text if spd-say is not available.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import threading
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Beat + Action dataclasses
# ---------------------------------------------------------------------------

# Antenna targets: [right, left] from the robot's perspective; range ≈ ±1.
# Positive = antenna up, negative = antenna down.


@dataclass
class Beat:
    """One antenna keyframe: move to *antennas* in *duration* seconds."""

    antennas: list[float]
    duration: float


@dataclass
class Action:
    """One recognised action: the label spoken aloud + the antenna reaction."""

    label: str          # spoken text
    beats: list[Beat]   # antenna choreography played while speaking
    pause_after: float = 1.0  # silence between actions (seconds)

    def total_duration(self) -> float:
        """Sum of beat durations + pause."""
        return sum(b.duration for b in self.beats) + self.pause_after


# ---------------------------------------------------------------------------
# Antenna choreographies — one per action class
# ---------------------------------------------------------------------------

def _beats_wave() -> list[Beat]:
    """Alternating left/right — mirrors a hand wave."""
    return [
        Beat([ 0.7, -0.2], 0.25),
        Beat([-0.2,  0.7], 0.25),
        Beat([ 0.7, -0.2], 0.25),
        Beat([-0.2,  0.7], 0.25),
        Beat([ 0.7, -0.2], 0.25),
        Beat([ 0.0,  0.0], 0.50),
    ]


def _beats_pushup() -> list[Beat]:
    """Slow rhythmic up-down — mimics the push-up cadence."""
    return [
        Beat([0.0, 0.0], 0.30),
        Beat([0.6, 0.6], 0.40),
        Beat([0.0, 0.0], 0.40),
        Beat([0.6, 0.6], 0.40),
        Beat([0.0, 0.0], 0.40),
        Beat([0.6, 0.6], 0.40),
        Beat([0.0, 0.0], 0.50),
    ]


def _beats_squat() -> list[Beat]:
    """Both antennas dip down then recover — echoes the squat motion."""
    return [
        Beat([-0.5, -0.5], 0.50),
        Beat([ 0.3,  0.3], 0.40),
        Beat([-0.5, -0.5], 0.50),
        Beat([ 0.3,  0.3], 0.40),
        Beat([ 0.0,  0.0], 0.50),
    ]


def _beats_clapping() -> list[Beat]:
    """Both antennas open wide then snap together — like clapping hands."""
    return [
        Beat([ 0.6, -0.6], 0.20),
        Beat([-0.6,  0.6], 0.20),
        Beat([ 0.6, -0.6], 0.20),
        Beat([-0.6,  0.6], 0.20),
        Beat([ 0.6, -0.6], 0.20),
        Beat([ 0.0,  0.0], 0.50),
    ]


def _beats_jumping_jacks() -> list[Beat]:
    """Fast alternating ups — energetic, like arms spreading in jumping jacks."""
    return [
        Beat([0.8,  0.8], 0.20),
        Beat([0.0,  0.0], 0.20),
        Beat([0.8,  0.8], 0.20),
        Beat([0.0,  0.0], 0.20),
        Beat([0.8,  0.8], 0.20),
        Beat([0.0,  0.0], 0.20),
        Beat([0.8,  0.8], 0.20),
        Beat([0.0,  0.0], 0.50),
    ]


def _beats_none() -> list[Beat]:
    """Idle — a slow, gentle sway: nothing detected."""
    return [
        Beat([ 0.2, -0.1], 0.60),
        Beat([-0.1,  0.2], 0.60),
        Beat([ 0.0,  0.0], 0.50),
    ]


# ---------------------------------------------------------------------------
# Action sequence
# ---------------------------------------------------------------------------

ACTIONS: list[Action] = [
    Action(
        label="No movement detected.",
        beats=_beats_none(),
        pause_after=1.0,
    ),
    Action(
        label="Wave detected.",
        beats=_beats_wave(),
        pause_after=1.0,
    ),
    Action(
        label="Push-up detected.",
        beats=_beats_pushup(),
        pause_after=1.0,
    ),
    Action(
        label="Squat detected.",
        beats=_beats_squat(),
        pause_after=1.0,
    ),
    Action(
        label="Clapping detected.",
        beats=_beats_clapping(),
        pause_after=1.0,
    ),
    Action(
        label="Jumping jacks detected.",
        beats=_beats_jumping_jacks(),
        pause_after=1.0,
    ),
]


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

def _find_tts() -> Callable[[str, int, str], None] | None:
    """Return a callable(text, rate, language) -> None, or None if unavailable."""
    if shutil.which("spd-say"):
        def _spd(text: str, rate: int, language: str) -> None:
            subprocess.run(
                ["spd-say", "--rate", str(rate), "--language", language, "--", text],
                check=False,
            )
        return _spd
    return None


def _speak(tts: Callable | None, text: str, rate: int, language: str) -> None:
    """Speak *text* via TTS (blocking) or fall back to printing."""
    if tts is not None:
        tts(text, rate, language)
    else:
        print(f"  [TTS unavailable] {text}")


def _announce_and_move(mini, tts, action: Action, rate: int, language: str) -> None:
    """Speak the action label while running the antenna choreography in parallel.

    The TTS runs in a background thread so the antenna beats execute
    concurrently in the main thread.
    """
    print(f'  → "{action.label}"')

    tts_thread = threading.Thread(
        target=_speak, args=(tts, action.label, rate, language), daemon=True
    )
    tts_thread.start()

    for beat in action.beats:
        mini.goto_target(antennas=beat.antennas, duration=beat.duration)

    tts_thread.join(timeout=12.0)
    time.sleep(action.pause_after)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="localhost",
                        help="daemon host (default: localhost)")
    parser.add_argument("--rate", type=int, default=-25,
                        help="spd-say speech rate -100..+100 (default: -25, slightly slower)")
    parser.add_argument("--language", default="en",
                        help="language code for spd-say (default: en)")
    parser.add_argument("--list-actions", action="store_true",
                        help="print the action list and exit")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = parse_args()

    if args.list_actions:
        print("Action  Label                       Duration")
        print("-" * 55)
        for i, a in enumerate(ACTIONS, 1):
            print(f"  {i}.  {a.label:<35} {a.total_duration():.1f}s")
        return

    tts = _find_tts()
    if tts is None:
        print("WARNING: spd-say not found — text will be printed, not spoken.")
        print("         sudo apt install speech-dispatcher")
        print()

    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    connection_mode = (
        "localhost_only" if args.host in ("localhost", "127.0.0.1") else "network"
    )
    print(f"Connecting to daemon on {args.host} ...")

    try:
        mini_ctx = ReachyMini(
            host=args.host,
            connection_mode=connection_mode,
            media_backend="no_media",
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not connect to daemon on {args.host}: {exc}\n"
            "Is sim/start.sh running?"
        ) from exc

    with mini_ctx as mini:
        print(f"Connected. Running {len(ACTIONS)} action announcements...\n")

        # Reset to neutral before starting.
        mini.goto_target(head=create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
        time.sleep(0.5)

        try:
            for i, action in enumerate(ACTIONS, 1):
                print(f"[{i}/{len(ACTIONS)}]", end=" ")
                _announce_and_move(mini, tts, action, args.rate, args.language)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            print("\nReturning antennas to neutral...")
            mini.goto_target(
                head=create_head_pose(), antennas=[0.0, 0.0], duration=1.0
            )

    print("Done.")


if __name__ == "__main__":
    main()
