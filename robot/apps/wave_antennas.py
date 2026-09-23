"""Wave recognition on the real Reachy Mini: the robot camera feeds the pose model, the antennas wave back.

Meant to run **on the robot** (daemon and camera are local, no window):

    ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.wave_antennas'

It also runs from this Mac (pose model here, frames over WebRTC, optional preview window):

    reachy_mini_env/bin/python -m robot.apps.wave_antennas --host 192.168.137.171 --window

Deployment to the robot: robot/deploy.sh (see docs/robot.md).
"""

import argparse
import math
import signal
import time

from core.frames import RobotCameraSource, downscale
from core.metrics import StageStats, format_summary
from core.motion.reaction import NEUTRAL_ANTENNAS
from core.motion.sender import TargetSender
from core.pose_backends import BACKENDS, create_backend

FRAME_TIMEOUT_S = 10.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost", help="daemon host (default: localhost = on the robot)")
    # MoveNet on TFLite/XNNPACK by default: ~3x faster than the ONNX build on the robot, and BlazePose
    # (mediapipe) aborts there entirely (CPU without AES extensions).
    parser.add_argument("--model", choices=list(BACKENDS), default="movenet-tflite", help="pose model")
    parser.add_argument("--wave-trigger", choices=["auto", "rules", "classifier"], default="auto",
                        help="detector the robot reacts to (auto: classifier if trained, else rules)")
    parser.add_argument("--classifier", default=None,
                        help="classifier to use (default: models/wave_classifier; the .npz next to it is what "
                             "actually runs). Try models/wave_classifier_ntu.joblib for the NTU-trained one")
    parser.add_argument("--min-score", type=float, default=None, help="keypoint confidence (default: per model)")
    parser.add_argument("--width", type=int, default=640, help="downscale frames to this width before inference")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="antenna updates per second (own thread; the daemon runs at 50 Hz)")
    parser.add_argument("--amplitude", type=float, default=20.0, help="antenna wave amplitude in degrees")
    parser.add_argument("--freq", type=float, default=1.5, help="antenna wave frequency in Hz")
    parser.add_argument("--wave-seconds", type=float, default=1.2, help="how long the antennas wave back")
    parser.add_argument("--cooldown", type=float, default=0.5,
                        help="dead time after a wave before another one is answered")
    parser.add_argument("--follow", action="store_true", help="also make the head follow the body center")
    parser.add_argument("--max-yaw", type=float, default=80.0, help="total gaze yaw limit in degrees")
    parser.add_argument("--max-head-offset", type=float, default=45.0,
                        help="how far the head may look away from the body (hardware limit 60 deg)")
    parser.add_argument("--body-gain", type=float, default=1.5,
                        help="how fast the body catches up with the gaze, per second (0 = body stays still)")
    parser.add_argument("--max-body-yaw", type=float, default=110.0, help="body rotation limit in degrees")
    parser.add_argument("--max-pitch", type=float, default=20.0, help="head pitch in degrees at the image edge")
    parser.add_argument("--deadzone", type=float, default=0.05, help="ignored offset from the image center")
    parser.add_argument("--follow-gain", type=float, default=120.0,
                        help="head speed in degrees per second per unit of image error")
    parser.add_argument("--window", action="store_true", help="show the camera window (needs a display)")
    parser.add_argument("--stream-port", type=int, default=0,
                        help="serve the annotated camera view at http://<robot>:PORT/ (0 = off)")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl+C)")
    parser.add_argument("--no-sleep", action="store_true", help="leave the robot awake on exit")
    return parser.parse_args()


def install_signal_handlers():
    """Make SIGTERM and SIGHUP behave like Ctrl+C, so the cleanup in main() still runs.

    Stopping this script over SSH sends SIGTERM, and a dropped SSH connection (Wi-Fi loss, closed laptop
    lid, sshd reaping the session) sends SIGHUP to the remote foreground process; both signals' default
    action skips the `finally` block and leaves the robot awake with its body off-center. Raising
    KeyboardInterrupt from the handler reuses the exact path Ctrl+C takes.
    """

    def handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGHUP, handler)
    return handler


def connect(host):
    """Preflight (with auto-fix) + an SDK connection with camera frames. Works on the robot and over the network."""
    from robot import preflight as reachy_preflight

    result = reachy_preflight.run_preflight(host=host, fix=True, needs=("motion", "media"))
    print(result.render())
    if not result.ready:
        raise SystemExit("Robot not ready (see the ✖ line above).")

    from reachy_mini import ReachyMini

    local = host in ("localhost", "127.0.0.1")
    return ReachyMini(
        host=result.host,
        connection_mode="localhost_only" if local else "network",
        media_backend="default",  # LOCAL camera socket on the robot, WebRTC from the Mac
        # The body yaw is commanded explicitly by the follower; the daemon's automatic assist would fight it.
        automatic_body_yaw=False,
    )


def follow_line(follower, error, body_measured):
    """Debug line for the overlay: where the gaze is and how it is split between head and body."""
    offset = "lost" if error is None else f"{error[0]:+.2f},{error[1]:+.2f}"
    return (f"err {offset}  gaze {follower.gaze_yaw:+.0f}  head {follower.yaw:+.0f}/{follower.pitch:+.0f}  "
            f"body {follower.body_yaw:+.0f} (is {body_measured:+.0f})", (255, 200, 0))


def annotate(frame, result, min_score, model_name, panel, panel_lines):
    """Draw the skeleton and the live panel on the frame (shared by the window and the MJPEG stream)."""
    from core.vision import draw_coco_skeleton, put_text

    if result.keypoints is not None:
        draw_coco_skeleton(frame, result.keypoints, min_score)
    put_text(frame, f"{model_name}  FPS {panel['fps']:.1f}", 1)
    for row, (text, color) in enumerate(panel_lines, start=2):
        put_text(frame, text, row, color)


def wait_for_frames(source, timeout_s=FRAME_TIMEOUT_S):
    """Return the first frame, or None if the camera stays silent (media released / tracker without frames)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = source.read()
        if frame is not None:
            return frame
        time.sleep(0.1)
    return None


def main():
    install_signal_handlers()
    args = parse_args()

    backend = create_backend(args.model)
    min_score = args.min_score if args.min_score is not None else backend.default_min_score

    from core.motion.live import WaveMonitor

    try:
        kwargs = {"model_path": args.classifier} if args.classifier else {}
        monitor = WaveMonitor(trigger=args.wave_trigger, min_score=min_score, **kwargs)
    except ValueError as exc:
        backend.close()
        raise SystemExit(str(exc))
    monitor.reaction.amplitude_deg = args.amplitude
    monitor.reaction.freq_hz = args.freq
    monitor.reaction.duration_s = args.wave_seconds
    monitor.reaction.cooldown_s = args.cooldown

    follower = None
    if args.follow:
        from core.tracking_math import CenteringFollower

        follower = CenteringFollower(
            gain=args.follow_gain, max_yaw=args.max_yaw, max_pitch=args.max_pitch, deadzone=args.deadzone,
            return_speed=args.follow_gain / 6, mirrored=False, body_gain=args.body_gain,
            max_head_offset=args.max_head_offset, max_body_yaw=args.max_body_yaw,
        )

    print(f"Model {backend.name} (min score {min_score}), trigger {monitor.trigger_name}, "
          f"antennas {args.amplitude:.0f} deg @ {args.freq:.1f} Hz, "
          f"wave {args.wave_seconds:.1f} s + {args.cooldown:.1f} s cooldown")
    stream = None
    if args.stream_port:
        from core.stream import MjpegServer

        stream = MjpegServer(port=args.stream_port).start()

    mini_context = connect(args.host)
    stats = StageStats()
    triggered = []

    with mini_context as mini:
        from reachy_mini.utils import create_head_pose

        source = RobotCameraSource(mini)
        print("Waiting for camera frames...")
        try:
            got_frames = wait_for_frames(source) is not None
        except KeyboardInterrupt:
            if not args.no_sleep:
                mini.goto_sleep()
            raise
        if not got_frames:
            if not args.no_sleep:
                mini.goto_sleep()
            raise SystemExit(
                "No camera frames. Check `curl -s http://%s:8000/api/media/status` "
                "(released=true means a client took the camera)." % args.host
            )
        print("Frames arriving. Ctrl+C to stop." if not args.window else "Frames arriving. ESC/q in the window to stop.")
        if stream is not None:
            print(f"Camera view (with the model overlay): http://<this robot>:{stream.port}/")
        mini.goto_target(antennas=list(NEUTRAL_ANTENNAS), duration=0.5)

        started = last_follow = last_follow_log = time.perf_counter()
        follow_debug = None
        # The antennas are driven by their own thread: the vision loop is far too slow (~6 FPS) to sample
        # the wave smoothly, and it only decides when a wave starts.
        sender = TargetSender(mini, monitor.reaction, rate_hz=args.rate).start()
        try:
            while True:
                frame = source.read()
                if frame is None:
                    continue
                frame = downscale(frame, args.width)

                result = backend.infer(frame)
                now = time.perf_counter()
                monitor.window.aspect_ratio = frame.shape[1] / frame.shape[0]
                before = monitor.reactions
                timings = {**result.timings, "motion": monitor.update(now, result.keypoints)}
                if monitor.reactions > before:
                    triggered.append(now - started)
                    print(f"[{now - started:6.1f}s] WAVE -> antennas")

                head = None
                if follower is not None:
                    from core.body_center import body_center

                    target = body_center(result.keypoints, min_score)
                    error = (target[0] - 0.5, target[1] - 0.5) if target else None
                    yaw, pitch = follower.update(error, now - last_follow)
                    last_follow = now
                    head = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
                    sender.body_yaw_deg = follower.body_yaw
                    if now - last_follow_log >= 1.0:
                        joints, _ = mini.get_current_joint_positions()
                        body_measured = math.degrees(joints[0])
                        follow_debug = follow_line(follower, error, body_measured)
                        print(f"[{now - started:6.1f}s] {follow_debug[0]}")
                        last_follow_log = now
                stats.add_frame(timings, has_target=result.keypoints is not None)

                sender.head = head

                if args.window or stream is not None:
                    lines = list(monitor.panel_lines(now))
                    if follow_debug is not None:
                        lines.append(follow_debug)
                    annotate(frame, result, min_score, backend.name, stats.window_summary(), lines)
                if stream is not None:
                    stream.update(frame)
                if args.window:
                    import cv2

                    cv2.imshow("Reachy Mini - wave recognition", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.seconds and now - started >= args.seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            sender.stop()
            if stream is not None:
                stream.close()
            print("Returning the antennas and the body to neutral...")
            mini.goto_target(head=create_head_pose(), antennas=list(NEUTRAL_ANTENNAS), body_yaw=0.0, duration=1.0)
            backend.close()
            if args.window:
                import cv2

                cv2.destroyAllWindows()
            if not args.no_sleep:
                mini.goto_sleep()
            print(format_summary(backend.name, stats.final_summary()))
            print(f"  wave detections: rules {monitor.triggers['rules']}, "
                  f"classifier {monitor.triggers['classifier']}; antennas answered {monitor.reactions} "
                  f"(trigger: {monitor.trigger_name}, the rest fell inside the cooldown)")
            if triggered:
                print("  reacted at: " + ", ".join(f"{t:.1f}s" for t in triggered))


if __name__ == "__main__":
    main()
