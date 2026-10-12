"""The coach in simulation: the robot teaches itself your exercises, then plays the workout with you.

    # terminal 1 - the MuJoCo simulator
    sim/start.sh

    # terminal 2 - the coach (stand 2-3 m back, whole body in view)
    reachy_mini_env/bin/python -m sim.coach --name Caio
    reachy_mini_env/bin/python -m sim.coach --name Caio --rounds 1 --round-seconds 4   # ~1 min dry run
    reachy_mini_env/bin/python -m sim.coach --name Caio --workout-only                 # reuse your model

The recording goes to collect/coach_recordings/<name>/ (keypoints only, gitignored), apart from the group's
dataset in collect/recordings/ so test runs never leak into it; the model trains on both folders and goes to
core/models/users/<name>.npz. ESC in the window stops; what was recorded is kept.
"""

import argparse
import time
from pathlib import Path

import cv2

from core.coach.pipeline import Pipeline
from core.coach.recording import GuidedRecorder, MIN_MOTION, MIN_VISIBLE, sanitize
from core.coach.trainer import Trainer
from core.motion.antennas import NEUTRAL_ANTENNAS
from core.motion.voice import DEFAULT_VOICE, ActionVoice
from core.pose_backends import BACKENDS, create_backend
from core.tracking import HeadFollower, body_center, image_error_to_angles
from core.vision import add_camera_argument, draw_coco_skeleton, open_camera, put_text

REPO = Path(__file__).resolve().parents[1]
GROUP_RECORDINGS = REPO / "collect" / "recordings"
WINDOW = "Reachy Mini - Coach"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="the child's name (spoken, and used in file names)")
    add_camera_argument(parser)
    parser.add_argument("--host", default="localhost", help="daemon host (default: local simulation)")
    # MoveNet: the pose model the group recorded with, so the child's features match theirs.
    parser.add_argument("--model", choices=list(BACKENDS), default="movenet-lightning", help="pose model")
    parser.add_argument("--min-score", type=float, default=None, help="keypoint confidence (default: per model)")
    parser.add_argument("--rounds", type=int, default=6, help="rounds per exercise (default 6)")
    parser.add_argument("--round-seconds", type=float, default=8.0, help="seconds per round (default 8)")
    parser.add_argument("--min-visible", type=float, default=MIN_VISIBLE, help="quality check: visible fraction")
    parser.add_argument("--min-motion", type=float, default=MIN_MOTION, help="quality check: torso lengths")
    parser.add_argument("--reps", type=int, default=3, help="workout reps per exercise (default 3)")
    parser.add_argument("--workout-only", action="store_true", help="skip recording, play with the saved model")
    parser.add_argument("--recordings", default=str(REPO / "collect" / "coach_recordings"),
                        help="where the coach saves (the group's collect/recordings is only read)")
    parser.add_argument("--models", default=str(REPO / "core" / "models" / "users"), help="per-child models")
    parser.add_argument("--voice-model", default=DEFAULT_VOICE, help="Piper voice")
    parser.add_argument("--max-yaw", type=float, default=40.0, help="head yaw at the image edge (deg)")
    parser.add_argument("--max-pitch", type=float, default=25.0, help="head pitch at the image edge (deg)")
    parser.add_argument("--deadzone", type=float, default=0.03, help="ignored offset from center (0..0.5)")
    parser.add_argument("--smoothing", type=float, default=0.2, help="head follow, 0..1, higher = faster")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    name = args.name.strip()
    if not sanitize(name):
        raise SystemExit("--name needs letters or digits")
    model_out = Path(args.models) / f"{sanitize(name)}.npz"
    if args.workout_only and not model_out.exists():
        raise SystemExit(f"No saved model at {model_out}; run without --workout-only first.")
    cap = open_camera(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("The camera gave no frame.")
    backend = create_backend(args.model)
    min_score = args.min_score if args.min_score is not None else backend.default_min_score
    follower = HeadFollower(smoothing=args.smoothing, return_speed=0.05)

    def follow(keypoints, now):
        target = body_center(keypoints, min_score)
        angles = None if target is None else image_error_to_angles(
            target[0], target[1], args.max_yaw, args.max_pitch, args.deadzone)
        return follower.update(angles)

    voice = ActionVoice(voice_model=args.voice_model)
    recorder = GuidedRecorder(frame.shape[1] / frame.shape[0], min_score, args.rounds, args.round_seconds,
                              min_visible=args.min_visible, min_motion=args.min_motion)
    trainer = Trainer([GROUP_RECORDINGS, args.recordings], model_out)
    connection = "localhost_only" if args.host in ("localhost", "127.0.0.1") else "network"
    with ReachyMini(host=args.host, connection_mode=connection, media_backend="no_media") as mini:
        mini.goto_target(head=create_head_pose(), antennas=list(NEUTRAL_ANTENNAS), duration=1.0)
        pipeline = Pipeline(mini, voice, name, recorder, trainer, args.recordings, backend.name, follow,
                            workout_reps=args.reps, workout_model=model_out if args.workout_only else None)
        pipeline.start()
        try:
            while not pipeline.finished:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                keypoints = backend.infer(frame).keypoints
                now = time.perf_counter()
                pipeline.update(now, keypoints)
                if keypoints is not None:
                    draw_coco_skeleton(frame, keypoints, min_score)
                for row, (text, color) in enumerate(pipeline.hud_lines(now, keypoints), start=1):
                    put_text(frame, text, row, color)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("Stopped.")
                    break
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            pipeline.close()
            mini.goto_target(head=create_head_pose(), antennas=list(NEUTRAL_ANTENNAS), duration=1.0)
            backend.close()
            cap.release()
            cv2.destroyAllWindows()
            voice.close()
    if pipeline.model_path is not None:
        print(f"Workout model: {pipeline.model_path}")
    if model_out.exists():
        print(f"Play again with it: reachy_mini_env/bin/python -m sim.coach --name {name} --workout-only")


if __name__ == "__main__":
    main()
