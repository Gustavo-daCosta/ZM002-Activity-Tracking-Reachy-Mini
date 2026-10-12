"""Stream the Reachy Mini camera and optionally run the workout session on the robot.

Standard remote action-control mode (default):
    python -m robot.remote_camera_server --host localhost
    Laptop posts antenna/head targets to /result.

Remote session mode:
    python -m robot.remote_camera_server --host localhost --session --width 192 --height 108
    Laptop streams MoveNet keypoints to /result; the SessionEngine, speech, gestures and
    torso-centering head/body tracking remain on the Reachy Mini. Following is enabled by default.

Endpoints:
    http://<robot-ip>:8765/stream.mjpg  camera MJPEG stream
    http://<robot-ip>:8766/result       POST JSON inference results/keypoints
    http://<robot-ip>:8766/health       status JSON
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from core.motion.antennas import NEUTRAL_ANTENNAS
from core.stream import MjpegServer
from core.tracking import CenteringFollower, body_center
from robot.apps.common import Picamera2Source, connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="Reachy daemon host (default: localhost)")
    parser.add_argument("--camera-port", type=int, default=8765)
    parser.add_argument("--result-port", type=int, default=8766)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument(
        "--session",
        action="store_true",
        help="run SessionEngine on the robot; /result must contain MoveNet keypoints",
    )
    parser.add_argument(
        "--activity-model",
        default=None,
        help=(
            "path to an exported activity-classifier .npz; defaults to the shared six-class model. "
            "For the personalized coach model, use ~/coach_data/models/<name>.npz and choose rounds "
            "that model knows (typically squat,arm_raise,side_bend)."
        ),
    )
    parser.add_argument("--rounds", default="squat,clapping,jumping_jacks,wave")
    parser.add_argument("--reps-per-round", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="during session mode, track the torso with the head/body (default: enabled; use --no-follow to disable)",
    )
    parser.add_argument("--follow-gain", type=float, default=120.0,
                        help="head tracking speed in degrees/s per image error")
    parser.add_argument("--max-yaw", type=float, default=80.0,
                        help="maximum total gaze yaw in degrees")
    parser.add_argument("--max-pitch", type=float, default=20.0,
                        help="maximum head pitch in degrees")
    parser.add_argument("--deadzone", type=float, default=0.05,
                        help="ignore smaller normalized image errors")
    parser.add_argument("--max-head-offset", type=float, default=45.0,
                        help="maximum head yaw offset from body yaw in degrees")
    parser.add_argument("--body-gain", type=float, default=1.5,
                        help="how quickly the body follows gaze (0 disables body rotation)")
    parser.add_argument("--max-body-yaw", type=float, default=110.0,
                        help="maximum body rotation in degrees")
    parser.add_argument("--voice-model", default="en_US-lessac-medium")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.session and (args.reps_per_round < 1 or args.timeout <= 0):
        parser.error("--reps-per-round must be >= 1 and --timeout must be positive")

    latest = {"received_at": 0.0, "payload": None, "version": 0}
    lock = threading.Lock()
    runtime = {
        "engine": None,
        "session_enabled": args.session,
        "follower": None,
        "monitor": None,
        "activity_model": None,
        "activity_classes": [],
    }

    class ResultHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/health", "/status"):
                self.send_error(404)
                return

            with lock:
                payload = latest["payload"]
                received_at = latest["received_at"]

            now = time.monotonic()
            body_data = {
                "status": "ok",
                "session_mode": args.session,
                "head_following": bool(args.session and args.follow),
                "inference_connected": payload is not None and now - received_at < 1.0,
                "model": payload.get("model") if isinstance(payload, dict) else None,
                "inference_fps": payload.get("fps") if isinstance(payload, dict) else None,
                "activity_model": runtime["activity_model"],
                "activity_classes": runtime["activity_classes"],
            }

            monitor_status = runtime.get("monitor")
            if monitor_status is not None and monitor_status.detection is not None:
                detection = monitor_status.detection
                body_data.update({
                    "detected_action": detection.action,
                    "detected_confidence": round(float(detection.score), 4),
                    "action_probabilities": {
                        str(name): round(float(score), 4)
                        for name, score in detection.probabilities.items()
                    },
                })

            follower_status = runtime.get("follower")
            if follower_status is not None:
                body_data.update({
                    "follow_head_yaw_deg": round(follower_status.yaw, 1),
                    "follow_head_pitch_deg": round(follower_status.pitch, 1),
                    "follow_body_yaw_deg": round(follower_status.body_yaw, 1),
                })

            engine = runtime["engine"]
            if args.session and engine is not None:
                cur = engine.current_round
                body_data.update({
                    "session_state": engine.state.name,
                    "round": engine.current_round_idx + 1 if cur is not None else None,
                    "rounds_total": len(engine.plan.rounds),
                    "action": cur.action if cur is not None else None,
                    "reps": engine.reps_this_round,
                    "reps_target": cur.target_reps if cur is not None else None,
                    "total_reps": engine.total_reps_completed,
                })

            body = json.dumps(body_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/result":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    self.send_error(413)
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                with lock:
                    latest["payload"] = payload
                    latest["received_at"] = time.monotonic()
                    latest["version"] += 1
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, *_args):
            pass

    result_server = ThreadingHTTPServer(("0.0.0.0", args.result_port), ResultHandler)
    result_server.daemon_threads = True
    result_thread = threading.Thread(target=result_server.serve_forever, daemon=True)
    result_thread.start()

    camera = None
    stream = None
    engine = None
    voice = None
    last_processed_version = 0
    last_session_update = time.perf_counter()
    last_status = 0.0
    follower = None
    last_follow_update = time.perf_counter()
    follow_was_active = False

    try:
        with connect(args.host) as mini:
            camera = Picamera2Source(width=args.width, height=args.height)
            stream = MjpegServer(port=args.camera_port, quality=75).start()
            print(f"Camera stream: http://<robot-ip>:{stream.port}/stream.mjpg")
            print(f"JSON result endpoint: http://<robot-ip>:{args.result_port}/result")
            print(f"Status endpoint: http://<robot-ip>:{args.result_port}/status")
            print(f"Camera resolution: {args.width}x{args.height}")

            if args.session:
                from core.motion.actions import DEFAULT_ACTION_MODEL, ActionDetector, ActionMonitor
                from core.motion.voice import ActionVoice
                from core.session.config import SessionPlan
                from core.session.engine import SessionEngine, SessionState
                from reachy_mini.utils import create_head_pose

                activity_model = (
                    Path(args.activity_model).expanduser()
                    if args.activity_model
                    else DEFAULT_ACTION_MODEL
                )
                try:
                    detector = ActionDetector(activity_model)
                except (OSError, ValueError) as exc:
                    raise SystemExit(f"Could not load activity model {activity_model}: {exc}") from exc
                if not detector.available:
                    raise SystemExit(
                        f"No activity model at {activity_model}. "
                        "Provide an exported .npz with --activity-model or ensure "
                        "core/models/action_classifier.npz exists on the robot."
                    )
                rounds = [item.strip() for item in args.rounds.split(",") if item.strip()]
                if not rounds:
                    raise SystemExit("--rounds must contain at least one action")
                unsupported = [action for action in rounds if action not in detector.classes or action == "none"]
                if unsupported:
                    raise SystemExit(
                        f"The activity model {activity_model} does not contain requested round(s): "
                        f"{', '.join(unsupported)}. Available classes: {', '.join(detector.classes)}. "
                        "Choose rounds supported by the selected model. The personalized coach model "
                        "typically supports squat,arm_raise,side_bend; it does not recognize "
                        "clapping,jumping_jacks,or wave."
                    )

                runtime["activity_model"] = str(activity_model.expanduser().resolve())
                runtime["activity_classes"] = list(detector.classes)
                print(f"Activity classifier: {runtime['activity_model']}")
                print(f"Activity classes: {', '.join(detector.classes)}")

                voice = ActionVoice(voice_model=args.voice_model)
                plan = SessionPlan.create(
                    exercises=rounds,
                    reps_per_round=args.reps_per_round,
                    timeout_s=args.timeout,
                )
                monitor = ActionMonitor(
                    aspect_ratio=args.width / args.height,
                    min_score=args.min_score,
                    detector=detector,
                    voice=False,
                )
                engine = SessionEngine(plan=plan, monitor=monitor, voice=voice, mini=mini)
                runtime["engine"] = engine
                runtime["monitor"] = monitor

                if args.follow:
                    follower = CenteringFollower(
                        gain=args.follow_gain,
                        max_yaw=args.max_yaw,
                        max_pitch=args.max_pitch,
                        deadzone=args.deadzone,
                        return_speed=args.follow_gain / 6.0,
                        mirrored=False,
                        body_gain=args.body_gain,
                        max_head_offset=args.max_head_offset,
                        max_body_yaw=args.max_body_yaw,
                    )
                    runtime["follower"] = follower
                    print(
                        "Head/body tracking ENABLED during active exercise rounds "
                        f"(gain={args.follow_gain:.0f}, body gain={args.body_gain:.2f}, "
                        f"limits: gaze ±{args.max_yaw:.0f}°, body ±{args.max_body_yaw:.0f}°)."
                    )
                else:
                    print("Head/body tracking disabled (--no-follow).")

                print(
                    f"REMOTE SESSION MODE: {len(plan.rounds)} rounds; "
                    f"{plan.reps_per_round} reps/round; timeout={args.timeout:.1f}s"
                )
                print("Waiting for the laptop to send a valid person pose; the workout starts on first detection.")
            else:
                from reachy_mini.utils import create_head_pose
                print("Remote action-control mode (not workout session mode).")

            print("Press Ctrl+C to stop.")

            while True:
                frame = camera.read()
                if frame is not None:
                    stream.update(frame)

                now_mono = time.monotonic()
                now_perf = time.perf_counter()
                with lock:
                    payload = latest["payload"]
                    received_at = latest["received_at"]
                    version = latest["version"]

                fresh = payload is not None and now_mono - received_at < 1.0

                if args.session:
                    # Parse the latest COCO-17 pose whenever the control loop runs. The
                    # SessionEngine is updated only on fresh POSTs, while the tracking
                    # controller can run smoothly using the most recent pose between POSTs.
                    keypoints = None
                    if fresh and isinstance(payload, dict):
                        raw_keypoints = payload.get("keypoints")
                        if raw_keypoints is not None:
                            try:
                                keypoints_array = np.asarray(raw_keypoints, dtype=np.float32)
                                if keypoints_array.shape == (17, 3):
                                    keypoints = keypoints_array
                                else:
                                    print(
                                        "Ignoring malformed keypoints shape:",
                                        keypoints_array.shape,
                                    )
                            except (TypeError, ValueError):
                                print("Ignoring malformed keypoints payload")

                    if version != last_processed_version:
                        if engine.state == SessionState.IDLE:
                            if keypoints is not None:
                                print("Valid pose received from laptop; starting workout session.")
                                engine.start()
                        else:
                            engine.update(now_perf, keypoints)
                        last_processed_version = version
                        last_session_update = now_perf
                    elif (
                        engine.state == SessionState.ACTIVE
                        and now_perf - last_session_update >= 0.25
                    ):
                        # Advance exercise timeouts even when the laptop stream is lost.
                        engine.update(now_perf, None)
                        last_session_update = now_perf

                    # Follow the torso only during ACTIVE exercise rounds. Pausing during
                    # introductions/feedback/outro prevents set_target() from fighting the
                    # SessionEngine's expressive goto_target() gestures.
                    if follower is not None:
                        if engine.state == SessionState.ACTIVE:
                            target = body_center(keypoints, args.min_score) if keypoints is not None else None
                            error = (target[0] - 0.5, target[1] - 0.5) if target else None
                            dt = min(max(now_perf - last_follow_update, 0.0), 0.1)
                            yaw, pitch = follower.update(error, dt)
                            mini.set_target(
                                head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True),
                                body_yaw=math.radians(follower.body_yaw),
                            )
                            follow_was_active = True
                        else:
                            if follow_was_active:
                                # Session gestures return the head/body to neutral. Reset the
                                # controller state so the next round resumes from that pose.
                                follower.gaze_yaw = 0.0
                                follower.body_yaw = 0.0
                                follower.yaw = 0.0
                                follower.pitch = 0.0
                            follow_was_active = False
                        last_follow_update = now_perf

                    if now_mono - last_status >= 2.0:
                        cur = engine.current_round
                        if fresh and isinstance(payload, dict):
                            print(
                                f"Laptop connected; model={payload.get('model', '?')} "
                                f"FPS={payload.get('fps', 0):.1f}; "
                                f"session={engine.state.name}; "
                                f"round={engine.current_round_idx + 1}/{len(engine.plan.rounds)}; "
                                f"reps={engine.reps_this_round}/"
                                f"{cur.target_reps if cur else 0}"
                                + (
                                    f"; follow=head {follower.yaw:+.0f}° / body {follower.body_yaw:+.0f}°"
                                    if follower is not None and engine.state == SessionState.ACTIVE
                                    else ""
                                )
                            )
                        else:
                            print(
                                f"Waiting for laptop keypoints; session={engine.state.name}"
                            )
                        last_status = now_mono
                else:
                    # Legacy mode: apply head/antenna/body targets computed on the laptop.
                    if fresh:
                        antennas = payload.get("antennas_rad") or list(NEUTRAL_ANTENNAS)
                        head_data = payload.get("head")
                        head = None
                        if isinstance(head_data, dict):
                            head = create_head_pose(
                                yaw=float(head_data.get("yaw_deg", 0.0)),
                                pitch=float(head_data.get("pitch_deg", 0.0)),
                                degrees=True,
                            )
                        body_yaw = payload.get("body_yaw_deg")
                        mini.set_target(
                            head=head,
                            antennas=[float(antennas[0]), float(antennas[1])],
                            body_yaw=(
                                None if body_yaw is None
                                else math.radians(float(body_yaw))
                            ),
                        )
                    else:
                        mini.set_target(
                            head=create_head_pose(),
                            antennas=list(NEUTRAL_ANTENNAS),
                            body_yaw=0.0,
                        )

                    if now_mono - last_status >= 2.0:
                        if fresh:
                            print(
                                f"Laptop inference connected; model={payload.get('model', '?')} "
                                f"FPS={payload.get('fps', 0):.1f}"
                            )
                        else:
                            print("Waiting for fresh laptop inference results...")
                        last_status = now_mono

                time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        if engine is not None:
            engine.close()
        if voice is not None:
            voice.close()
        if stream is not None:
            stream.close()
        if camera is not None:
            camera.close()
        result_server.shutdown()
        result_server.server_close()


if __name__ == "__main__":
    main()
