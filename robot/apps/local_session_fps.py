"""Run the exercise session locally on the Reachy Mini and measure MoveNet FPS.

This is a standalone on-robot comparison to laptop-side remote_session_inference.
It uses MoveNet Lightning INT8 via LiteRT/TFLite on the Reachy's CPU and captures
192x108 frames directly using Picamera2. The activity classifier and workout
SessionEngine are the same components used by the rest of this repository.

Default workout:
    squat -> arm_raise -> side_bend
    4 repetitions per round, 60-second timeout per round.

Run on the Reachy Mini:
    ~/camera_env/bin/python -m robot.apps.local_session_fps

Stop with Ctrl+C at any time. The script prints live FPS and the average FPS
for the complete measured run at the end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from core.motion.actions import ActionDetector, ActionMonitor
from core.motion.antennas import NEUTRAL_ANTENNAS
from core.motion.voice import ActionVoice, DEFAULT_VOICE
from core.pose_backends import create_backend
from core.session.config import SessionPlan
from core.session.engine import SessionEngine, SessionState
from core.vision import downscale
from robot.apps.common import Picamera2Source, connect, wait_for_frames

DEFAULT_ROUNDS = "squat,arm_raise,side_bend"
DEFAULT_REPS_PER_ROUND = 4
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_ACTIVITY_MODEL = Path.home() / "coach_data" / "models" / "tester.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rounds",
        default=DEFAULT_ROUNDS,
        help=f"comma-separated exercise rounds (default: {DEFAULT_ROUNDS})",
    )
    parser.add_argument(
        "--reps-per-round",
        type=int,
        default=DEFAULT_REPS_PER_ROUND,
        help=f"target repetitions per round (default: {DEFAULT_REPS_PER_ROUND})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"maximum seconds per round (default: {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--activity-model",
        type=Path,
        default=DEFAULT_ACTIVITY_MODEL,
        help=f"trained activity classifier .npz (default: {DEFAULT_ACTIVITY_MODEL})",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=192,
        help="Picamera2 output width (default: 192; matches laptop stream)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=108,
        help="Picamera2 output height (default: 108; matches laptop stream)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="maximum frame width passed to MoveNet (default: 192)",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=1.0,
        help="seconds between live FPS reports (default: 1.0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="inference warm-up runs excluded from FPS statistics (default: 8)",
    )
    parser.add_argument(
        "--voice-model",
        default=DEFAULT_VOICE,
        help=f"Piper voice model (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="leave the robot awake after the session",
    )
    return parser.parse_args()


def _state_changed(state: SessionState, detail: str) -> None:
    print(f"\n[SESSION] {state.name}: {detail}", flush=True)


def _print_summary(
    started: float | None,
    ended: float | None,
    frame_count: int,
    detected_count: int,
    pose_ms: list[float],
    session_ms: list[float],
) -> None:
    if started is None or ended is None or frame_count == 0:
        print("\nNo measured camera frames; average FPS is unavailable.")
        return

    elapsed = max(ended - started, 1e-9)
    print("\n" + "=" * 60)
    print("REACHY MINI LOCAL MOVENET FPS SUMMARY")
    print("=" * 60)
    print("Pose model: MoveNet Lightning INT8 (LiteRT/TFLite, Reachy CPU)")
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time:     {elapsed:.2f} s")
    print(f"Average FPS:      {frame_count / elapsed:.2f}")
    print(f"Person detected:  {100.0 * detected_count / frame_count:.1f}% of frames")
    if pose_ms:
        print(
            "Pose inference:   "
            f"mean {np.mean(pose_ms):.2f} ms; "
            f"p95 {np.percentile(pose_ms, 95):.2f} ms"
        )
    if session_ms:
        print(
            "Session update:   "
            f"mean {np.mean(session_ms):.2f} ms; "
            f"p95 {np.percentile(session_ms, 95):.2f} ms"
        )
    print("=" * 60)


def main() -> None:
    args = parse_args()

    if args.reps_per_round < 1:
        raise SystemExit("--reps-per-round must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")
    if min(args.camera_width, args.camera_height, args.width) < 1:
        raise SystemExit("Camera dimensions and --width must be positive")
    if args.report_interval <= 0:
        raise SystemExit("--report-interval must be greater than 0")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")

    exercises = [part.strip() for part in args.rounds.split(",") if part.strip()]
    if not exercises:
        raise SystemExit("At least one exercise must be supplied in --rounds")

    activity_model = args.activity_model.expanduser().resolve()
    if not activity_model.is_file():
        raise SystemExit(
            f"Activity model not found: {activity_model}\n"
            "Pass --activity-model with a trained .npz model. The default expects "
            "the Coach-trained model at ~/coach_data/models/tester.npz."
        )

    plan = SessionPlan.create(
        exercises=exercises,
        reps_per_round=args.reps_per_round,
        timeout_s=args.timeout,
    )

    started: float | None = None
    ended: float | None = None
    frame_count = 0
    detected_count = 0
    pose_times: list[float] = []
    session_times: list[float] = []

    backend = None
    source = None
    voice = None
    engine = None
    monitor = None

    print("Connecting to Reachy Mini...", flush=True)
    try:
        with connect("localhost") as mini:
            try:
                print(
                    "Loading MoveNet Lightning INT8 with LiteRT/TFLite on the robot...",
                    flush=True,
                )
                backend = create_backend("movenet-tflite")

                detector = ActionDetector(activity_model)
                if not detector.available:
                    raise SystemExit(f"Could not load activity model: {activity_model}")

                print(f"Activity classifier: {activity_model}", flush=True)
                print("Activity classes: " + ", ".join(detector.classes), flush=True)
                missing = sorted(set(exercises) - set(detector.classes))
                if missing:
                    raise SystemExit(
                        "The activity model does not contain the requested round(s): "
                        f"{', '.join(missing)}. Available classes: {', '.join(detector.classes)}"
                    )

                # Use the same 192x108 source dimensions as the laptop MJPEG stream
                # to make the comparison as close as possible and avoid a large frame.
                source = Picamera2Source(
                    width=args.camera_width,
                    height=args.camera_height,
                )
                first_frame = wait_for_frames(source)
                if first_frame is None:
                    raise SystemExit("No frames received from Picamera2")

                frame = downscale(first_frame, args.width)
                aspect_ratio = frame.shape[1] / frame.shape[0]
                min_score = backend.default_min_score
                print(
                    f"Direct camera: {frame.shape[1]}x{frame.shape[0]} "
                    f"(configured {args.camera_width}x{args.camera_height})",
                    flush=True,
                )
                print(
                    f"Warming up MoveNet ({args.warmup} runs; excluded from FPS)...",
                    flush=True,
                )
                for _ in range(args.warmup):
                    backend.infer(frame)

                monitor = ActionMonitor(
                    aspect_ratio=aspect_ratio,
                    min_score=min_score,
                    detector=detector,
                    voice=False,
                )
                voice = ActionVoice(voice_model=args.voice_model)

                from reachy_mini.utils import create_head_pose

                mini.goto_target(
                    head=create_head_pose(),
                    antennas=list(NEUTRAL_ANTENNAS),
                    body_yaw=0.0,
                    duration=1.0,
                )
                time.sleep(0.3)

                engine = SessionEngine(
                    plan=plan,
                    monitor=monitor,
                    voice=voice,
                    mini=mini,
                    on_state_change=_state_changed,
                )

                print("\n" + "=" * 60, flush=True)
                print("STARTING LOCAL REACHY SESSION FPS TEST", flush=True)
                print("Pose backend: movenet-tflite (LiteRT/TFLite on Reachy CPU)", flush=True)
                print(f"Rounds: {', '.join(exercises)}", flush=True)
                print(
                    f"Target: {args.reps_per_round} reps/round; "
                    f"timeout: {args.timeout:.0f}s/round",
                    flush=True,
                )
                print(
                    f"Model input source: {frame.shape[1]}x{frame.shape[0]}; "
                    "MoveNet tensor: 192x192",
                    flush=True,
                )
                print(
                    "Live lines show interval FPS and running average FPS. "
                    "Press Ctrl+C to stop and get a final summary.",
                    flush=True,
                )
                print("=" * 60 + "\n", flush=True)

                engine.start()
                started = time.perf_counter()
                last_report = started
                interval_frames = 0

                try:
                    while not engine.is_finished:
                        raw_frame = source.read()
                        if raw_frame is None:
                            time.sleep(0.005)
                            continue

                        frame = downscale(raw_frame, args.width)
                        infer_start = time.perf_counter()
                        result = backend.infer(frame)
                        now = time.perf_counter()
                        pose_elapsed_ms = result.timings.get(
                            "pose",
                            (now - infer_start) * 1000.0,
                        )

                        # Always update the session, including frames with no keypoints,
                        # so round timeouts still elapse if the person leaves the frame.
                        session_start = time.perf_counter()
                        engine.update(now, result.keypoints)
                        session_elapsed_ms = (time.perf_counter() - session_start) * 1000.0

                        frame_count += 1
                        interval_frames += 1
                        detected_count += int(result.keypoints is not None)
                        pose_times.append(float(pose_elapsed_ms))
                        session_times.append(session_elapsed_ms)

                        report_now = time.perf_counter()
                        interval_elapsed = report_now - last_report
                        if interval_elapsed >= args.report_interval:
                            interval_fps = interval_frames / interval_elapsed
                            average_fps = frame_count / max(report_now - started, 1e-9)
                            current_round = engine.current_round
                            if current_round is None:
                                exercise_info = "exercise=complete"
                            else:
                                exercise_info = (
                                    f"exercise={current_round.action}; "
                                    f"reps={engine.reps_this_round}/{current_round.target_reps}"
                                )
                            print(
                                f"FPS={interval_fps:.1f}; average={average_fps:.1f}; "
                                f"frames={frame_count}; pose={pose_elapsed_ms:.1f}ms; "
                                f"keypoints={'yes' if result.keypoints is not None else 'none'}; "
                                f"state={engine.state.name}; {exercise_info}",
                                flush=True,
                            )
                            last_report = report_now
                            interval_frames = 0

                except KeyboardInterrupt:
                    print("\nInterrupted by user; calculating FPS summary...", flush=True)

            finally:
                # Capture the benchmark end time before camera/model/robot cleanup so
                # shutdown and sleep movement do not distort the reported average FPS.
                if started is not None:
                    ended = time.perf_counter()

                if engine is not None:
                    try:
                        engine.close()
                    except Exception:
                        pass
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:
                        pass
                if voice is not None:
                    try:
                        voice.close()
                    except Exception:
                        pass

                try:
                    from reachy_mini.utils import create_head_pose

                    mini.goto_target(
                        head=create_head_pose(),
                        antennas=list(NEUTRAL_ANTENNAS),
                        body_yaw=0.0,
                        duration=1.0,
                    )
                    if not args.no_sleep:
                        mini.goto_sleep()
                except Exception as exc:
                    print(
                        f"Cleanup warning: could not return robot to neutral/sleep: {exc}",
                        flush=True,
                    )

    finally:
        _print_summary(
            started,
            ended,
            frame_count,
            detected_count,
            pose_times,
            session_times,
        )


if __name__ == "__main__":
    main()
