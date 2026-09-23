"""Multi-action recognition on the real Reachy Mini: the robot camera feeds the pose model, the antennas
answer each recognized action with its own signature.

Meant to run **on the robot** (daemon and camera are local, no window):

    ssh reachy 'cd ~/wave_app && ~/wave_env/bin/python -m robot.apps.action_recognition'

It also runs from this Mac (pose model here, frames over WebRTC, optional preview window):

    reachy_mini_env/bin/python -m robot.apps.action_recognition --host 192.168.137.171 --window

**Place the robot so its camera sees the whole person** -- at the edge of the table, or stand further back.
Waves and clapping survive a waist-up framing; a squat misses about 1 repetition in 7 with the legs in frame
and about 1 in 3 waist-up (recall 0.856 / 0.685), and push-ups do not work at all. Measured per-class F1 and
what is honestly demonstrable: docs/action-recognition.md.

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost", help="daemon host (default: localhost = on the robot)")
    # MoveNet on TFLite/XNNPACK by default: ~3x faster than the ONNX build on the robot, and BlazePose
    # (mediapipe) aborts there entirely (CPU without AES extensions).
    parser.add_argument("--model", choices=list(BACKENDS), default="movenet-tflite", help="pose model")
    parser.add_argument("--classifier", default=None,
                        help="action model to run (default: core/models/action_classifier.npz; only the "
                             ".npz runs on the robot, which has no scikit-learn)")
    parser.add_argument("--min-score", type=float, default=None, help="keypoint confidence (default: per model)")
    parser.add_argument("--width", type=int, default=640, help="downscale frames to this width before inference")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="antenna updates per second (own thread; the daemon runs at 50 Hz)")
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
    return parser.parse_args(argv)


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


def frame_aspect_ratio(frame):
    """width / height of the frame the pose model actually sees.

    Never a constant. `normalize_to_torso` multiplies x by this before measuring anything, and the training
    windows used each clip's own width/height, so a wrong ratio rescales every horizontal quantity and,
    through the joint chains, every angle: a live path fixed at 16/9 shifted `elbow_angle_amplitude` by up
    to 17.2 degrees on 340x256 clips and flipped the predicted class in a quarter of the jumping-jack
    windows. Called *after* `downscale`, on the frame that is handed to the backend.
    """
    return frame.shape[1] / frame.shape[0]


class MonitorReaction:
    """Adapter that lets one `TargetSender` sample whichever action reaction is running.

    `TargetSender` samples `reaction.angles(now)` and adds `NEUTRAL_ANTENNAS` itself, while
    `ActionMonitor.antennas(now)` returns the already-offset pose. So this returns the **raw** offsets of
    the active reaction (None when none is), which makes `antennas_with_neutral(self.angles(now))` equal to
    `monitor.antennas(now)` frame for frame -- `tests/test_action_recognition.py` asserts exactly that.
    Adding the neutral pose here as well would double it and drive the antennas out of range.

    `ActionMonitor.note` never starts a reaction while another one is running, so at most one is ever
    active and the sampled target is continuous.
    """

    def __init__(self, monitor):
        self._monitor = monitor

    def angles(self, now):
        for reaction in self._monitor.reactions.values():
            if reaction.active(now):
                return reaction.angles(now)
        return None


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

    from core.motion.actions.detector import DEFAULT_ACTION_MODEL, ActionDetector
    from core.motion.actions.live import ActionMonitor

    # Fail before touching the robot when the model is missing: ActionMonitor can only be built once the
    # first frame has given us the real aspect ratio, by which point the robot is already awake.
    model_path = args.classifier or DEFAULT_ACTION_MODEL
    # Built once and handed to the ActionMonitor below: re-reading the 5.3 MB .npz costs real time on the CM4.
    detector = ActionDetector(model_path)
    if not detector.available:
        backend.close()
        raise SystemExit(f"No action model at {model_path}. Deploy it with robot/deploy.sh "
                         "--code-only, or train it with `python -m training.actions.train`.")

    follower = None
    if args.follow:
        from core.tracking_math import CenteringFollower

        follower = CenteringFollower(
            gain=args.follow_gain, max_yaw=args.max_yaw, max_pitch=args.max_pitch, deadzone=args.deadzone,
            return_speed=args.follow_gain / 6, mirrored=False, body_gain=args.body_gain,
            max_head_offset=args.max_head_offset, max_body_yaw=args.max_body_yaw,
        )

    print(f"Model {backend.name} (min score {min_score}), action model {model_path}")
    print("Place the robot so the camera sees the WHOLE person: squats and push-ups need the legs in frame.")
    stream = None
    if args.stream_port:
        from core.stream import MjpegServer

        stream = MjpegServer(port=args.stream_port).start()

    mini_context = connect(args.host)
    stats = StageStats()
    recognized = []

    with mini_context as mini:
        from reachy_mini.utils import create_head_pose

        # Everything after this point is inside the try/finally: the robot is awake from here on, and a
        # SIGHUP (dropped SSH link) arriving while the monitor is being built, while `goto_target` blocks or
        # while the sender starts must still reach `goto_sleep`. `sender` and `monitor` may therefore be
        # unbound when the cleanup runs, so both start as None and the cleanup guards them.
        sender = None
        monitor = None
        try:
            source = RobotCameraSource(mini)
            print("Waiting for camera frames...")
            first = wait_for_frames(source)
            if first is None:
                raise SystemExit(
                    "No camera frames. Check `curl -s http://%s:8000/api/media/status` "
                    "(released=true means a client took the camera)." % args.host
                )
            aspect_ratio = frame_aspect_ratio(downscale(first, args.width))
            monitor = ActionMonitor(aspect_ratio, min_score=min_score, detector=detector)
            print(f"Frames arriving ({aspect_ratio:.3f} aspect ratio). "
                  + ("ESC/q in the window to stop." if args.window else "Ctrl+C to stop."))
            if stream is not None:
                print(f"Camera view (with the model overlay): http://<this robot>:{stream.port}/")
            mini.goto_target(antennas=list(NEUTRAL_ANTENNAS), duration=0.5)

            started = last_follow = last_follow_log = time.perf_counter()
            follow_debug = None
            # The antennas are driven by their own thread: the vision loop is far too slow (~10 FPS) to
            # sample a reaction smoothly, and it only decides which action starts one.
            sender = TargetSender(mini, MonitorReaction(monitor), rate_hz=args.rate).start()
            while True:
                frame = source.read()
                if frame is None:
                    continue
                frame = downscale(frame, args.width)

                result = backend.infer(frame)
                now = time.perf_counter()
                # Re-derived every frame: the camera can renegotiate its resolution mid-run.
                monitor.window.aspect_ratio = frame_aspect_ratio(frame)
                before = dict(monitor.counts)
                timings = {**result.timings, "motion": monitor.update(now, result.keypoints)}
                for action, count in monitor.counts.items():
                    if count > before[action]:
                        recognized.append((now - started, action))
                        print(f"[{now - started:6.1f}s] {action.upper()} recognized")

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
                    # panel_lines is a generator: materialize it once, never len() it.
                    lines = list(monitor.panel_lines(now))
                    if follow_debug is not None:
                        lines.append(follow_debug)
                    annotate(frame, result, min_score, backend.name, stats.window_summary(), lines)
                if stream is not None:
                    stream.update(frame)
                if args.window:
                    import cv2

                    cv2.imshow("Reachy Mini - action recognition", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.seconds and now - started >= args.seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if sender is not None:
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
            # These are *recognitions*, not antenna answers: an occurrence detected while another reaction
            # is still swinging is counted here but deliberately left unanswered.
            counts = ""
            if monitor is not None:
                counts = " ".join(f"{action}:{count}" for action, count in monitor.counts.items() if count)
            print(f"  actions recognized: {counts or '-'}")
            if recognized:
                print("  recognized at: " + ", ".join(f"{t:.1f}s {action}" for t, action in recognized))


if __name__ == "__main__":
    main()
