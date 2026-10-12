"""The coach on the robot: Reachy Mini learns a child's exercises on its own CPU, then plays the workout.

On the robot (camera through Picamera2, view the camera at http://<robot>:8080/):
    ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.coach --name Ana --stream-port 8080'

Data stays on the robot: recordings (keypoints only) in ~/coach_data/recordings/<name>/, models in
~/coach_data/models/<name>.npz. robot/deploy.sh copies the group's recordings there. Ctrl+C stops; what was
recorded is kept. Deployment: robot/deploy.sh.
"""

import time
from pathlib import Path

from core.coach.pipeline import Pipeline
from core.coach.recording import GuidedRecorder, MIN_MOTION, MIN_VISIBLE, sanitize
from core.coach.trainer import Trainer
from core.motion.antennas import NEUTRAL_ANTENNAS
from core.motion.voice import DEFAULT_VOICE, ActionVoice
from core.pose_backends import create_backend
from core.stream import MjpegServer
from core.tracking import CenteringFollower, body_center
from core.vision import downscale, draw_coco_skeleton, put_text
from robot.apps.common import Picamera2Source, connect, make_parser, wait_for_frames

DATA = Path.home() / "coach_data"


def parse_args(argv=None):
    parser = make_parser(__doc__)
    parser.add_argument("--name", required=True, help="the child's name (spoken, and used in file names)")
    parser.add_argument("--rounds", type=int, default=6, help="rounds per exercise (default 6)")
    parser.add_argument("--round-seconds", type=float, default=8.0, help="seconds per round (default 8)")
    parser.add_argument("--min-visible", type=float, default=MIN_VISIBLE, help="quality check: visible fraction")
    parser.add_argument("--min-motion", type=float, default=MIN_MOTION, help="quality check: torso lengths")
    parser.add_argument("--reps", type=int, default=3, help="workout reps per exercise (default 3)")
    parser.add_argument("--workout-only", action="store_true", help="skip recording, play with the saved model")
    parser.add_argument("--recordings", default=str(DATA / "recordings"), help="recordings folder")
    parser.add_argument("--models", default=str(DATA / "models"), help="per-child models")
    parser.add_argument("--voice-model", default=DEFAULT_VOICE, help="Piper voice")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    name = args.name.strip()
    if not sanitize(name):
        raise SystemExit("--name needs letters or digits")
    model_out = Path(args.models) / f"{sanitize(name)}.npz"
    if args.workout_only and not model_out.exists():
        raise SystemExit(f"No saved model at {model_out}; run without --workout-only first.")
    backend = create_backend(args.model)
    min_score = args.min_score if args.min_score is not None else backend.default_min_score
    follower = CenteringFollower(gain=args.follow_gain, max_yaw=args.max_yaw, max_pitch=args.max_pitch,
                                 deadzone=args.deadzone, return_speed=args.follow_gain / 6, mirrored=False,
                                 max_head_offset=args.max_head_offset)
    last = [None]

    def follow(keypoints, now):
        target = body_center(keypoints, min_score)
        error = None if target is None else (target[0] - 0.5, target[1] - 0.5)
        dt = 0.0 if last[0] is None else now - last[0]
        last[0] = now
        return follower.update(error, dt)

    stream = MjpegServer(port=args.stream_port).start() if args.stream_port else None
    with connect(args.host) as mini:
        from reachy_mini.utils import create_head_pose

        if args.host in ("localhost", "127.0.0.1"):
            source = Picamera2Source(width=1920, height=1080)
        else:
            from core.vision import RobotCameraSource
            source = RobotCameraSource()
        first = wait_for_frames(source)
        if first is None:
            raise SystemExit("No camera frames.")
        first = downscale(first, args.width)
        voice = ActionVoice(voice_model=args.voice_model)
        recorder = GuidedRecorder(first.shape[1] / first.shape[0], min_score, args.rounds, args.round_seconds,
                                  min_visible=args.min_visible, min_motion=args.min_motion)
        trainer = Trainer(args.recordings, model_out)
        mini.goto_target(head=create_head_pose(), antennas=list(NEUTRAL_ANTENNAS), body_yaw=0.0, duration=1.0)
        pipeline = Pipeline(mini, voice, name, recorder, trainer, args.recordings, backend.name, follow,
                            workout_reps=args.reps, workout_model=model_out if args.workout_only else None,
                            rate_hz=args.rate)
        if stream is not None:
            print(f"Camera view: http://<this robot>:{stream.port}/")
        pipeline.start()
        try:
            while not pipeline.finished:
                frame = source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                frame = downscale(frame, args.width)
                keypoints = backend.infer(frame).keypoints
                now = time.perf_counter()
                pipeline.update(now, keypoints)
                if stream is not None:
                    if keypoints is not None:
                        draw_coco_skeleton(frame, keypoints, min_score)
                    for row, (text, color) in enumerate(pipeline.hud_lines(now, keypoints), start=1):
                        put_text(frame, text, row, color)
                    stream.update(frame)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            pipeline.close()
            source.close()
            if stream is not None:
                stream.close()
            backend.close()
            mini.goto_target(head=create_head_pose(), antennas=list(NEUTRAL_ANTENNAS), body_yaw=0.0,
                             duration=1.0)
            if not args.no_sleep:
                mini.goto_sleep()
            voice.close()
    if pipeline.model_path is not None:
        print(f"Workout model: {pipeline.model_path}")


if __name__ == "__main__":
    main()
