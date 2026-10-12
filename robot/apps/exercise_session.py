"""Interactive Reachy Mini Exercise Session on the physical robot.

A guided workout session where the robot acts as an interactive coach:
- Sequences through exercise rounds (configurable reps per round, defaulting to 1).
- Filters for the target exercise only, ignoring other movements.
- Real-time repetition feedback: counts reps aloud ("1... 2... 3...") with energetic antenna beats.
- Friendly, expressive gestures combining head nods, tilts, and antenna choreographies
  during greetings, round transitions, and celebrations.

Run directly on the robot (local camera, no window):
    ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.exercise_session --reps-per-round 1'

Run from this computer over network (WebRTC video, optional browser stream):
    reachy_mini_env/bin/python -m robot.apps.exercise_session --host 192.168.137.171 --stream-port 8080

Deployment: robot/deploy.sh
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
from core.stream import MjpegServer
from core.vision import downscale, draw_coco_skeleton, put_text
from robot.apps.common import Picamera2Source, make_parser, wait_for_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = make_parser(__doc__)
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
    return parser.parse_args(argv)


def draw_hud(frame, engine: SessionEngine, result, min_score: float) -> None:
    """Draw exercise session HUD on frame for stream/window display."""
    if result.keypoints is not None:
        draw_coco_skeleton(frame, result.keypoints, min_score)

    state = engine.state
    put_text(frame, f"REACHY MINI SESSION - {state.name}", 1, (0, 255, 0))

    cur = engine.current_round
    if cur is not None and state in (SessionState.ACTIVE, SessionState.ROUND_INTRO):
        put_text(
            frame,
            f"Round {engine.current_round_idx + 1}/{len(engine.plan.rounds)}: {cur.action.upper()}",
            2,
            (255, 255, 255),
        )
        put_text(
            frame,
            f"Reps: {engine.reps_this_round} / {cur.target_reps}",
            3,
            (0, 255, 255),
        )
        if state == SessionState.ACTIVE:
            elapsed = time.perf_counter() - engine.round_start_time
            remaining = max(0.0, cur.timeout_s - elapsed)
            put_text(frame, f"Time: {remaining:.1f}s", 4, (200, 200, 200))
    elif state == SessionState.ROUND_FEEDBACK:
        put_text(frame, "Round Finished! Great job!", 2, (0, 255, 0))
    elif state in (SessionState.INITIALIZING, SessionState.IDLE):
        put_text(frame, "Preparing workout...", 2, (255, 200, 0))
    elif state == SessionState.SESSION_OUTRO:
        put_text(
            frame,
            f"Workout Complete! Total reps: {engine.total_reps_completed}",
            2,
            (0, 255, 0),
        )


def main() -> None:
    """Entry point for robot exercise session app."""
    args = parse_args()

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
            f"No action model at {DEFAULT_ACTION_MODEL}. Deploy it with robot/deploy.sh."
        )

    print(f"Workout Plan: {len(plan.rounds)} rounds, {plan.reps_per_round} reps per round.")
    print("Connecting to robot...")

    from robot.apps.common import connect

    stream = None
    if args.stream_port is not None:
        stream = MjpegServer(port=args.stream_port).start()

    with connect(args.host) as mini:
        from reachy_mini.utils import create_head_pose

        voice = ActionVoice(voice_model=args.voice_model)
        backend = create_backend(args.model)
        min_score = (
            args.min_score
            if args.min_score is not None
            else backend.default_min_score
        )

        source = None
        try:
            # Initialize camera source
            if args.host in ("localhost", "127.0.0.1"):
                source = Picamera2Source(width=1920, height=1080)
            else:
                from core.vision import RobotCameraSource
                source = RobotCameraSource()
        except Exception as exc:
            print(f"Camera init error: {exc}. Falling back to SDK media feed if available.")

        # Set initial neutral pose
        mini.goto_target(
            head=create_head_pose(),
            antennas=list(NEUTRAL_ANTENNAS),
            body_yaw=0.0,
            duration=1.0,
        )
        time.sleep(0.5)

        first_frame = None
        if source is not None:
            first_frame = wait_for_frames(source)

        aspect_ratio = (
            first_frame.shape[1] / first_frame.shape[0]
            if first_frame is not None
            else 4.0 / 3.0
        )

        monitor = ActionMonitor(
            aspect_ratio=aspect_ratio,
            min_score=min_score,
            detector=detector,
            voice=False,  # Engine drives voice directly
        )

        engine = SessionEngine(
            plan=plan,
            monitor=monitor,
            voice=voice,
            mini=mini,
        )

        print("\n" + "=" * 60)
        print(f"  STARTING ROBOT EXERCISE SESSION: {len(plan.rounds)} rounds, {plan.reps_per_round} reps/round")
        print("=" * 60 + "\n")

        # Begin session: speaks welcome greeting + plays welcoming gesture
        engine.start()

        try:
            while not engine.is_finished:
                frame = source.read() if source is not None else None
                if frame is None:
                    time.sleep(0.02)
                    continue

                frame = downscale(frame, args.width)
                now = time.perf_counter()

                # Infer pose and update engine
                result = backend.infer(frame)
                if result.keypoints is not None:
                    engine.update(now, result.keypoints)

                # Render stream / window
                draw_hud(frame, engine, result, min_score)

                if stream is not None:
                    stream.update(frame)

                if args.window:
                    cv2.imshow("Reachy Mini - Exercise Session", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("\nSession aborted by user.")
                        break

        except KeyboardInterrupt:
            print("\nSession interrupted.")
        finally:
            print("\nReturning robot to neutral and cleaning up...")
            if stream is not None:
                stream.close()
            if source is not None:
                source.close()
            backend.close()
            if args.window:
                cv2.destroyAllWindows()

            mini.goto_target(
                head=create_head_pose(),
                antennas=list(NEUTRAL_ANTENNAS),
                body_yaw=0.0,
                duration=1.0,
            )

            if not args.no_sleep:
                mini.goto_sleep()

            voice.close()

    print("Exercise session complete!")


if __name__ == "__main__":
    main()
