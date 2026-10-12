"""Stream the Reachy Mini camera and optionally run the workout session on the robot.

Standard remote action-control mode (default):
    python -m robot.remote_camera_server --host localhost
    Laptop posts antenna/head targets to /result.

Remote session mode:
    python -m robot.remote_camera_server --host localhost --session --width 192 --height 108
    Laptop streams MoveNet keypoints to /result; the SessionEngine, speech and robot
    gestures remain on the Reachy Mini.

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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from core.motion.antennas import NEUTRAL_ANTENNAS
from core.stream import MjpegServer
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
    parser.add_argument("--rounds", default="squat,clapping,jumping_jacks,wave")
    parser.add_argument("--reps-per-round", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--voice-model", default="en_US-lessac-medium")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.session and (args.reps_per_round < 1 or args.timeout <= 0):
        parser.error("--reps-per-round must be >= 1 and --timeout must be positive")

    latest = {"received_at": 0.0, "payload": None, "version": 0}
    lock = threading.Lock()
    runtime = {"engine": None, "session_enabled": args.session}

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
                "inference_connected": payload is not None and now - received_at < 1.0,
                "model": payload.get("model") if isinstance(payload, dict) else None,
                "inference_fps": payload.get("fps") if isinstance(payload, dict) else None,
            }

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

                detector = ActionDetector(DEFAULT_ACTION_MODEL)
                if not detector.available:
                    raise SystemExit(
                        f"No action model at {DEFAULT_ACTION_MODEL}. "
                        "Ensure core/models/action_classifier.npz exists on the robot."
                    )
                rounds = [item.strip() for item in args.rounds.split(",") if item.strip()]
                if not rounds:
                    raise SystemExit("--rounds must contain at least one action")

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
                    if version != last_processed_version:
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
