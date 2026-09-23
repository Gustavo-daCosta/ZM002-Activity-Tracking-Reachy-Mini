"""Body tracking on the simulated Reachy Mini, with interchangeable pose models for comparison.

The laptop webcam stands in for the robot camera. The selected pose model finds the person, the robot
follows the center of the torso (fallback: shoulders, then nose), and the camera window shows the model
output with live timings. A summary is printed on exit.

    # terminal 1
    sim/start.sh
    # terminal 2
    reachy_mini_env/bin/python -m sim.body_tracking --model blazepose-lite
    reachy_mini_env/bin/python -m sim.body_tracking --model vitpose-s
    reachy_mini_env/bin/python -m sim.body_tracking --detect-wave   # + wave recognition (antennas wave back)
    reachy_mini_env/bin/python -m sim.body_tracking --detect-actions  # + six-class action recognition

`--detect-actions` recognizes wave, clapping, squat, jumping jacks and (nominally) push-ups, each with its
own antenna signature. Stand back far enough for your legs to be in frame: squats and push-ups have no
usable signal without them. The camera is selected by NAME (`--camera FaceTime`), never by index, because
OpenCV's indices move when an iPhone connects through Continuity Camera.
"""

import argparse
import time

import cv2

from core.body_center import body_center
from core.metrics import StageStats, format_summary
from core.pose_backends import BACKENDS, create_backend
from core.tracking_math import HeadFollower, image_error_to_angles
from core.vision import add_camera_argument, draw_coco_skeleton, open_camera, put_text


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(BACKENDS), default="blazepose-lite", help="pose model")
    add_camera_argument(parser)
    parser.add_argument("--host", default="localhost", help="daemon host (default: local simulation)")
    parser.add_argument("--rate", type=float, default=20.0, help="max head updates per second")
    parser.add_argument("--max-yaw", type=float, default=40.0, help="degrees at the image edge")
    parser.add_argument("--max-pitch", type=float, default=25.0, help="degrees at the image edge")
    parser.add_argument("--deadzone", type=float, default=0.03, help="ignored offset from center (0..0.5)")
    parser.add_argument("--smoothing", type=float, default=0.2, help="0..1, higher = faster")
    parser.add_argument("--min-score", type=float, default=None, help="keypoint confidence (default: per model)")
    parser.add_argument("--detect-wave", action="store_true", help="recognize hand waves; antennas wave back")
    parser.add_argument("--wave-trigger", choices=["auto", "rules", "classifier"], default="auto",
                        help="detector that makes the robot react (auto: classifier if trained, else rules)")
    parser.add_argument("--detect-actions", action="store_true",
                        help="recognize six actions (none/wave/pushup/squat/clapping/jumping_jacks); the "
                             "antennas answer each one with its own signature. Needs the whole body in frame")
    parser.add_argument("--action-model", default=None,
                        help="action model to run (default: core/models/action_classifier.npz)")
    return parser.parse_args(argv)


def draw_overlay(frame, result, target, min_score, panel, model, yaw, pitch, extra_lines=()):
    h, w = frame.shape[:2]
    if result.box is not None:
        x0, y0, x1, y1 = result.box
        cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 128, 0), 2)
    confident = 0
    if result.keypoints is not None:
        draw_coco_skeleton(frame, result.keypoints, min_score)
        confident = int((result.keypoints[:, 2] >= min_score).sum())
    if target is not None:
        cv2.drawMarker(frame, (int(target[0] * w), int(target[1] * h)), (0, 255, 255), cv2.MARKER_CROSS, 24, 3)

    stages = "  ".join(f"{stage} {ms:.0f}ms" for stage, ms in panel["ms"].items())
    put_text(frame, f"{model}  FPS {panel['fps']:.1f}", 1)
    put_text(frame, stages or "-", 2)
    put_text(frame, f"keypoints {confident}/17  target {target[2] if target else 'lost'}", 3,
             (0, 255, 0) if target else (0, 0, 255))
    put_text(frame, f"yaw {yaw:+.1f}  pitch {pitch:+.1f}", 4, (0, 255, 0))
    for row, (text, color) in enumerate(extra_lines, start=5):
        put_text(frame, text, row, color)


def main():
    args = parse_args()

    # Import the SDK late so --help works instantly.
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    cap = open_camera(args.camera)
    backend = create_backend(args.model)
    min_score = args.min_score if args.min_score is not None else backend.default_min_score
    follower = HeadFollower(smoothing=args.smoothing, return_speed=0.05)
    stats = StageStats()
    update_interval = 1.0 / args.rate

    monitor = None
    action_model = None
    if args.detect_wave and args.detect_actions:
        backend.close()
        cap.release()
        raise SystemExit("--detect-wave and --detect-actions both drive the antennas: pick one.")
    if args.detect_wave:
        from core.motion.live import WaveMonitor

        try:
            monitor = WaveMonitor(trigger=args.wave_trigger)
        except ValueError as exc:
            backend.close()
            cap.release()
            raise SystemExit(str(exc))
        print(f"Wave recognition on (trigger: {monitor.trigger_name})")
    if args.detect_actions:
        # The ActionMonitor itself is built on the first frame: its aspect ratio must be the camera's real
        # one and open_camera's requested 640x480 is only a request. Checking the model here keeps the
        # "no model" failure out of the loop.
        from core.motion.actions.detector import DEFAULT_ACTION_MODEL, ActionDetector

        action_model = args.action_model or DEFAULT_ACTION_MODEL
        # Built once and reused by the ActionMonitor below, so the .npz is parsed a single time.
        action_detector = ActionDetector(action_model)
        if not action_detector.available:
            backend.close()
            cap.release()
            raise SystemExit(f"No action model at {action_model}. Train it with "
                             "`python -m training.actions.train --ntu ... --ucf ...`.")
        print(f"Action recognition on ({action_model}). Stand back far enough for your legs to be in frame.")

    print(f"Model {backend.name} (min score {min_score}). Connecting to the daemon on {args.host}...")
    connection_mode = "localhost_only" if args.host in ("localhost", "127.0.0.1") else "network"
    try:
        mini_context = ReachyMini(host=args.host, connection_mode=connection_mode, media_backend="no_media")
    except Exception as exc:
        backend.close()
        cap.release()
        raise SystemExit(f"Could not connect to the daemon on {args.host}: {exc}\nIs sim/start.sh running?")

    with mini_context as mini:
        print("Connected. ESC or q to quit.")
        mini.goto_target(head=create_head_pose(), duration=1.0)
        last_update = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read a frame.")
                    break
                frame = cv2.flip(frame, 1)  # mirror: behaves like looking at the robot
                if action_model is not None and monitor is None:
                    from core.motion.actions.live import ActionMonitor

                    monitor = ActionMonitor(frame.shape[1] / frame.shape[0], min_score=min_score,
                                            detector=action_detector)
                    print(f"Action window using the camera's own aspect ratio: "
                          f"{frame.shape[1]}x{frame.shape[0]} = {monitor.window.aspect_ratio:.3f}")

                result = backend.infer(frame)
                target = body_center(result.keypoints, min_score)
                angles = None
                if target is not None:
                    angles = image_error_to_angles(target[0], target[1], args.max_yaw, args.max_pitch, args.deadzone)
                yaw, pitch = follower.update(angles)  # no target: drift back to center

                now = time.perf_counter()
                timings = result.timings
                if monitor is not None:
                    monitor.window.aspect_ratio = frame.shape[1] / frame.shape[0]
                    timings = {**result.timings, "motion": monitor.update(now, result.keypoints)}
                stats.add_frame(timings, has_target=target is not None)

                if now - last_update >= update_interval:
                    antennas = monitor.antennas(now) if monitor is not None else None
                    mini.set_target(head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True), antennas=antennas)
                    last_update = now

                # ActionMonitor.panel_lines is a generator: materialize it, never iterate it twice.
                extra_lines = list(monitor.panel_lines(now)) if monitor is not None else ()
                draw_overlay(frame, result, target, min_score, stats.window_summary(), backend.name, yaw, pitch,
                             extra_lines)
                cv2.imshow("Reachy Mini sim - body tracking", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            print("Returning head to center...")
            mini.goto_target(head=create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
            backend.close()
            cap.release()
            cv2.destroyAllWindows()
            print(format_summary(backend.name, stats.final_summary()))
            if monitor is not None and args.detect_wave:
                print(f"  wave triggers: rules {monitor.triggers['rules']}, classifier {monitor.triggers['classifier']}"
                      f" (robot reacted to {monitor.trigger_name})")
            elif monitor is not None:
                # Recognitions, not antenna answers: an occurrence detected while another reaction is still
                # swinging is counted but deliberately left unanswered.
                counts = " ".join(f"{action}:{count}" for action, count in monitor.counts.items() if count)
                print(f"  actions recognized: {counts or '-'}")


if __name__ == "__main__":
    main()
