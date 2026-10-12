"""Interactive Exercise Session in simulation (webcam + MuJoCo 3D robot).

Runs a guided workout session where Reachy Mini coaches the user through a sequence
of exercises with targeted recognition, real-time rep counting ("1... 2... 3..."),
and expressive head and antenna movements for greetings, cheers, and celebrations.

Prerequisites:
    # terminal 1 — start the MuJoCo simulator
    sim/start.sh

    # terminal 2 — run the exercise session
    reachy_mini_env/bin/python -m sim.exercise_session [--reps-per-round 1]

Options:
    --reps-per-round REPS   repetitions required per round (default: 1)
    --rounds ROUNDS         comma-separated action sequence (default: squat,clapping,jumping_jacks,wave)
    --timeout TIMEOUT       seconds allowed per round before moving on (default: 30.0)
    --voice-model MODEL     Piper voice model (default: en_US-lessac-medium)
    --model MODEL           pose model: blazepose-lite, blazepose-full, vitpose-s
    --camera CAMERA         webcam name (default: FaceTime; see camera_check --list)
    --host HOST             simulation daemon host (default: localhost)
    --no-window             run headless without OpenCV display
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from core.motion.actions import DEFAULT_ACTION_MODEL, ActionDetector, ActionMonitor
from core.motion.antennas import NEUTRAL_ANTENNAS
from core.motion.voice import ActionVoice, DEFAULT_VOICE
from core.pose_backends import BACKENDS, create_backend
from core.session.config import DEFAULT_REPS_PER_ROUND, DEFAULT_TIMEOUT_S, SessionPlan
from core.session.engine import SessionEngine, SessionState
from core.vision import add_camera_argument, draw_coco_skeleton, open_camera, put_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reps-per-round",
        type=int,
        default=DEFAULT_REPS_PER_ROUND,
        help=f"number of reps required per exercise round (default: {DEFAULT_REPS_PER_ROUND})",
    )
    parser.add_argument(
        "--rounds",
        default="squat,clapping,jumping_jacks,wave",
        help="comma-separated actions for the workout rounds (default: squat,clapping,jumping_jacks,wave)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"seconds allowed per exercise round (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--voice-model",
        default=DEFAULT_VOICE,
        help=f"Piper voice model name (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--model",
        choices=list(BACKENDS),
        default="blazepose-lite",
        help="pose model (default: blazepose-lite)",
    )
    add_camera_argument(parser)
    parser.add_argument(
        "--host",
        default="localhost",
        help="daemon host (default: localhost)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="disable OpenCV preview window",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="keypoint confidence threshold",
    )
    return parser.parse_args(argv)


def draw_session_overlay(
    frame,
    engine: SessionEngine,
    result,
    min_score: float,
) -> None:
    """Draw session HUD: state, target exercise, progress, skeleton, and timers."""
    h, w = frame.shape[:2]

    # Draw skeleton
    if result.keypoints is not None:
        draw_coco_skeleton(frame, result.keypoints, min_score)

    # State banner
    state = engine.state
    banner_color = (0, 255, 0) if state == SessionState.ACTIVE else (0, 255, 255)
    put_text(frame, f"REACHY MINI WORKOUT - {state.name}", 1, banner_color)

    cur = engine.current_round
    if cur is not None and state in (SessionState.ACTIVE, SessionState.ROUND_INTRO):
        round_info = f"Round {engine.current_round_idx + 1}/{len(engine.plan.rounds)}: {cur.action.upper()}"
        put_text(frame, round_info, 2, (255, 255, 255))

        rep_progress = f"Reps: {engine.reps_this_round} / {cur.target_reps}"
        put_text(frame, rep_progress, 3, (0, 255, 255))

        if state == SessionState.ACTIVE:
            elapsed = time.perf_counter() - engine.round_start_time
            remaining = max(0.0, cur.timeout_s - elapsed)
            put_text(frame, f"Time remaining: {remaining:.1f}s", 4, (200, 200, 200))
    elif state == SessionState.ROUND_FEEDBACK:
        put_text(frame, f"Round Complete! Next round coming up...", 2, (0, 255, 0))
    elif state in (SessionState.INITIALIZING, SessionState.IDLE):
        put_text(frame, "Get ready! Session starting...", 2, (255, 200, 0))
    elif state == SessionState.SESSION_OUTRO:
        put_text(
            frame,
            f"Workout Complete! Total reps: {engine.total_reps_completed}",
            2,
            (0, 255, 0),
        )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Parse round exercises
    exercises = [e.strip() for e in args.rounds.split(",") if e.strip()]
    if not exercises:
        raise SystemExit("Error: at least one exercise must be provided in --rounds.")

    plan = SessionPlan.create(
        exercises=exercises,
        reps_per_round=args.reps_per_round,
        timeout_s=args.timeout,
    )

    detector = ActionDetector(DEFAULT_ACTION_MODEL)
    if not detector.available:
        raise SystemExit(
            f"No action model at {DEFAULT_ACTION_MODEL}. Deploy or train it first."
        )

    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    connection_mode = (
        "localhost_only" if args.host in ("localhost", "127.0.0.1") else "network"
    )
    print(f"Connecting to Reachy Mini on {args.host} ...")

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

    cap = open_camera(args.camera)
    backend = create_backend(args.model)
    min_score = (
        args.min_score
        if args.min_score is not None
        else backend.default_min_score
    )

    # Initialize ActionVoice and ActionMonitor
    voice = ActionVoice(voice_model=args.voice_model)
    aspect_ratio = 4.0 / 3.0  # reasonable default until camera frame reads
    monitor = ActionMonitor(aspect_ratio=aspect_ratio, min_score=min_score, detector=detector)

    with mini_ctx as mini:
        print(f"Connected to Reachy Mini. Setting neutral pose...")
        mini.goto_target(
            head=create_head_pose(),
            antennas=list(NEUTRAL_ANTENNAS),
            duration=1.0,
        )
        time.sleep(0.5)

        engine = SessionEngine(
            plan=plan,
            monitor=monitor,
            voice=voice,
            mini=mini,
        )

        print("\n" + "=" * 60)
        print(f"  STARTING EXERCISE SESSION: {len(plan.rounds)} rounds, {plan.reps_per_round} reps/round")
        print("=" * 60 + "\n")

        # Start workout with welcome speech and gesture
        engine.start()

        try:
            while not engine.is_finished:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                now = time.perf_counter()

                # Infer human pose
                result = backend.infer(frame)

                # Process frame in session engine
                if result.keypoints is not None:
                    engine.update(now, result.keypoints)

                # Preview window
                if not args.no_window:
                    draw_session_overlay(frame, engine, result, min_score)
                    cv2.imshow("Reachy Mini - Exercise Session", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("\nSession aborted by user.")
                        break

        except KeyboardInterrupt:
            print("\nSession interrupted.")
        finally:
            print("\nReturning robot to neutral...")
            mini.goto_target(
                head=create_head_pose(),
                antennas=list(NEUTRAL_ANTENNAS),
                duration=1.0,
            )
            backend.close()
            cap.release()
            if not args.no_window:
                cv2.destroyAllWindows()
            voice.close()

    print("\nWorkout session finished. Great job!")


if __name__ == "__main__":
    main()
