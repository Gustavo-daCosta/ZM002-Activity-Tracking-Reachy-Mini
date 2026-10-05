"""The camera -> pose -> monitor -> antennas loop shared by the recognition apps."""

import argparse
import math
import signal
import time

from core.metrics import StageStats, format_summary
from core.motion.antennas import NEUTRAL_ANTENNAS, TargetSender
from core.pose_backends import BACKENDS, create_backend
from core.stream import MjpegServer
from core.tracking import CenteringFollower, body_center
from core.vision import downscale, draw_coco_skeleton, put_text

FRAME_TIMEOUT_S = 10.0

class Picamera2Source:
    """Direct IMX708 camera access through libcamera/Picamera2."""

    def __init__(self, width=1920, height=1080):
        from picamera2 import Picamera2

        self._camera = Picamera2()

        config = self._camera.create_video_configuration(
            main={
                "size": (width, height),
                "format": "RGB888",
            }
        )

        self._camera.configure(config)
        self._camera.start()

        time.sleep(1.0)

    def read(self):
        try:
            # RGB888 is intentionally used without a channel swap.
            # Picamera2/libcamera presents the resulting array in the
            # BGR byte order expected by OpenCV.
            return self._camera.capture_array()
        except Exception:
            return None

    def close(self):
        if self._camera is None:
            return

        try:
            self._camera.stop()
        except Exception:
            pass

        try:
            self._camera.close()
        except Exception:
            pass

        self._camera = None

def make_parser(doc):
    """Argparse parser with the options common to both apps.

    Args:
        doc: The app's module docstring, shown by `--help`.
    """
    parser = argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="daemon host (default: localhost = on the robot)",
    )

    parser.add_argument(
        "--model",
        choices=list(BACKENDS),
        default="movenet-tflite",
        help="pose model",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="keypoint confidence (default: per model)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="downscale frames to this width before inference",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="antenna updates per second (own thread; the daemon runs at 50 Hz)",
    )

    parser.add_argument(
        "--follow",
        action="store_true",
        help="also make the head follow the body center",
    )

    parser.add_argument(
        "--max-yaw",
        type=float,
        default=80.0,
        help="total gaze yaw limit in degrees",
    )

    parser.add_argument(
        "--max-head-offset",
        type=float,
        default=45.0,
        help="how far the head may look away from the body (hardware limit 60 deg)",
    )

    parser.add_argument(
        "--body-gain",
        type=float,
        default=1.5,
        help="how fast the body catches up with the gaze, per second (0 = body stays still)",
    )

    parser.add_argument(
        "--max-body-yaw",
        type=float,
        default=110.0,
        help="body rotation limit in degrees",
    )

    parser.add_argument(
        "--max-pitch",
        type=float,
        default=20.0,
        help="head pitch in degrees at the image edge",
    )

    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.05,
        help="ignored offset from the image center",
    )

    parser.add_argument(
        "--follow-gain",
        type=float,
        default=120.0,
        help="head speed in degrees per second per unit of image error",
    )

    parser.add_argument(
        "--window",
        action="store_true",
        help="show the camera window (needs a display)",
    )

    parser.add_argument(
        "--stream-port",
        type=int,
        default=0,
        help="serve the annotated camera view at http://<robot>:PORT/ (0 = off)",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="stop after N seconds (0 = until Ctrl+C)",
    )

    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="leave the robot awake on exit",
    )

    return parser


def install_signal_handlers():
    """Make SIGTERM (ssh stop) and SIGHUP (dropped ssh link) behave like Ctrl+C."""

    def handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGHUP, handler)

    return handler


def connect(host):
    """Preflight + SDK connection without claiming the camera.

    Camera frames are obtained directly through Picamera2/libcamera.
    """

    from robot import preflight as reachy_preflight

    # IMPORTANT:
    # Do not request "media" here.
    # The camera belongs to the direct Picamera2 pipeline.
    result = reachy_preflight.run_preflight(
        host=host,
        fix=True,
        needs=("motion",),
    )

    print(result.render())

    if not result.ready:
        raise SystemExit(
            "Robot not ready (see the ✖ line above)."
        )

    from reachy_mini import ReachyMini

    local = host in ("localhost", "127.0.0.1")

    return ReachyMini(
        host=result.host,
        connection_mode=(
            "localhost_only"
            if local
            else "network"
        ),
        media_backend="no_media",
        automatic_body_yaw=False,
    )


def follow_line(follower, error, body_measured):
    """Overlay line: where the gaze is and how it is split between head and body."""

    offset = (
        "lost"
        if error is None
        else f"{error[0]:+.2f},{error[1]:+.2f}"
    )

    return (
        f"err {offset}  "
        f"gaze {follower.gaze_yaw:+.0f}  "
        f"head {follower.yaw:+.0f}/{follower.pitch:+.0f}  "
        f"body {follower.body_yaw:+.0f} "
        f"(is {body_measured:+.0f})",
        (255, 200, 0),
    )


def annotate(
    frame,
    result,
    min_score,
    model_name,
    panel,
    panel_lines,
):
    """Draw the skeleton and the live panel on the frame."""

    if result.keypoints is not None:
        draw_coco_skeleton(
            frame,
            result.keypoints,
            min_score,
        )

    put_text(
        frame,
        f"{model_name}  FPS {panel['fps']:.1f}",
        1,
    )

    for row, (text, color) in enumerate(
        panel_lines,
        start=2,
    ):
        put_text(
            frame,
            text,
            row,
            color,
        )


def wait_for_frames(
    source,
    timeout_s=FRAME_TIMEOUT_S,
):
    """Return the first frame, or None if the camera stays silent."""

    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        frame = source.read()

        if frame is not None:
            return frame

        time.sleep(0.1)

    return None


def run(
    args,
    make_monitor,
    events,
    summary,
    window_title,
):
    """Run the recognition loop until Ctrl+C, `--seconds` or a key in the window.

    Args:
        args: Parsed options from `make_parser`.
        make_monitor: `f(aspect_ratio, min_score) -> monitor` with `window`,
            `update(now, keypoints)`, `antennas(now)` and `panel_lines(now)`.
            Called after the first frame arrives.
        events: `f(monitor) -> {label: count}`; a count that grows prints
            `label` with a timestamp.
        summary: `f(monitor, recognized)` printing the app's final lines;
            `recognized` is a list of `(seconds, label)`.
        window_title: Title of the preview window.
    """

    install_signal_handlers()

    backend = create_backend(args.model)

    min_score = (
        args.min_score
        if args.min_score is not None
        else backend.default_min_score
    )

    print(
        f"Model {backend.name} "
        f"(min score {min_score})"
    )

    follower = None

    if args.follow:
        follower = CenteringFollower(
            gain=args.follow_gain,
            max_yaw=args.max_yaw,
            max_pitch=args.max_pitch,
            deadzone=args.deadzone,
            return_speed=args.follow_gain / 6,
            mirrored=False,
            body_gain=args.body_gain,
            max_head_offset=args.max_head_offset,
            max_body_yaw=args.max_body_yaw,
        )

    stream = None

    if args.stream_port:
        stream = MjpegServer(
            port=args.stream_port
        ).start()

    stats = StageStats()
    recognized = []

    with connect(args.host) as mini:
        from reachy_mini.utils import create_head_pose

        sender = None
        monitor = None
        source = None

        try:
            # Direct camera access.
            # The Reachy SDK media pipeline is NOT used.
            source = Picamera2Source(
                width=1920,
                height=1080,
            )

            print(
                "Direct camera: Picamera2/libcamera "
                "1920x1080"
            )

            print("Waiting for camera frames...")

            first = wait_for_frames(source)

            if first is None:
                raise SystemExit(
                    "No camera frames from Picamera2/libcamera."
                )

            first = downscale(
                first,
                args.width,
            )

            aspect_ratio = (
                first.shape[1] /
                first.shape[0]
            )

            monitor = make_monitor(
                aspect_ratio,
                min_score,
            )

            print(
                f"Frames arriving ({aspect_ratio:.3f} aspect ratio). "
                + (
                    "ESC/q in the window to stop."
                    if args.window
                    else "Ctrl+C to stop."
                )
            )

            if stream is not None:
                print(
                    "Camera view (with the model overlay): "
                    f"http://<this robot>:{stream.port}/"
                )

            mini.goto_target(
                antennas=list(NEUTRAL_ANTENNAS),
                duration=0.5,
            )

            started = time.perf_counter()
            last_follow = started
            last_follow_log = started

            follow_debug = None

            sender = TargetSender(
                mini,
                monitor.antennas,
                rate_hz=args.rate,
            ).start()

            while True:
                frame = source.read()

                if frame is None:
                    continue

                frame = downscale(
                    frame,
                    args.width,
                )

                result = backend.infer(frame)

                now = time.perf_counter()

                monitor.window.aspect_ratio = (
                    frame.shape[1] /
                    frame.shape[0]
                )

                before = events(monitor)

                timings = {
                    **result.timings,
                    "motion": monitor.update(
                        now,
                        result.keypoints,
                    ),
                }

                current_events = events(monitor)

                for label, count in current_events.items():
                    if count > before.get(
                        label,
                        0,
                    ):
                        recognized.append(
                            (
                                now - started,
                                label,
                            )
                        )

                        print(
                            f"[{now - started:6.1f}s] "
                            f"{label}"
                        )

                head = None

                if follower is not None:
                    target = body_center(
                        result.keypoints,
                        min_score,
                    )

                    error = (
                        (
                            target[0] - 0.5,
                            target[1] - 0.5,
                        )
                        if target
                        else None
                    )

                    yaw, pitch = follower.update(
                        error,
                        now - last_follow,
                    )

                    last_follow = now

                    head = create_head_pose(
                        yaw=yaw,
                        pitch=pitch,
                        degrees=True,
                    )

                    sender.body_yaw_deg = (
                        follower.body_yaw
                    )

                    if (
                        now - last_follow_log
                        >= 1.0
                    ):
                        joints, _ = (
                            mini.get_current_joint_positions()
                        )

                        follow_debug = follow_line(
                            follower,
                            error,
                            math.degrees(joints[0]),
                        )

                        print(
                            f"[{now - started:6.1f}s] "
                            f"{follow_debug[0]}"
                        )

                        last_follow_log = now

                stats.add_frame(
                    timings,
                    has_target=(
                        result.keypoints
                        is not None
                    ),
                )

                sender.head = head

                if (
                    args.window
                    or stream is not None
                ):
                    lines = list(
                        monitor.panel_lines(now)
                    )

                    if follow_debug is not None:
                        lines.append(
                            follow_debug
                        )

                    annotate(
                        frame,
                        result,
                        min_score,
                        backend.name,
                        stats.window_summary(),
                        lines,
                    )

                if stream is not None:
                    stream.update(frame)

                if args.window:
                    import cv2

                    cv2.imshow(
                        window_title,
                        frame,
                    )

                    if (
                        cv2.waitKey(1) & 0xFF
                        in (
                            27,
                            ord("q"),
                        )
                    ):
                        break

                if (
                    args.seconds
                    and now - started
                    >= args.seconds
                ):
                    break

        except KeyboardInterrupt:
            pass

        finally:
            if sender is not None:
                sender.stop()

            if source is not None:
                source.close()

            if stream is not None:
                stream.close()

            print(
                "Returning the antennas and the body "
                "to neutral..."
            )

            mini.goto_target(
                head=create_head_pose(),
                antennas=list(NEUTRAL_ANTENNAS),
                body_yaw=0.0,
                duration=1.0,
            )

            backend.close()

            if args.window:
                import cv2

                cv2.destroyAllWindows()

            if not args.no_sleep:
                mini.goto_sleep()

            print(
                format_summary(
                    backend.name,
                    stats.final_summary(),
                )
            )

            if monitor is not None:
                summary(
                    monitor,
                    recognized,
                )
