"""Serve the direct Picamera2 camera to a laptop and apply its JSON inference results.

Run on the Reachy Mini itself:
    python -m robot.remote_camera_server --host localhost

Laptop endpoints:
    http://<robot-ip>:8765/stream.mjpg  (camera)
    http://<robot-ip>:8766/result       (POST inference JSON)
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.motion.antennas import NEUTRAL_ANTENNAS
from core.stream import MjpegServer
from robot.apps.common import Picamera2Source, connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="Reachy daemon host (default: localhost)")
    parser.add_argument("--camera-port", type=int, default=8765)
    parser.add_argument("--result-port", type=int, default=8766)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    latest = {"received_at": 0.0, "payload": None}
    lock = threading.Lock()

    class ResultHandler(BaseHTTPRequestHandler):
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
    try:
        with connect(args.host) as mini:
            from reachy_mini.utils import create_head_pose

            camera = Picamera2Source(width=args.width, height=args.height)
            stream = MjpegServer(port=args.camera_port, quality=75).start()
            print(f"Camera stream: http://<robot-ip>:{stream.port}/stream.mjpg")
            print(f"JSON result endpoint: http://<robot-ip>:{args.result_port}/result")
            print("Press Ctrl+C to stop.")
            last_status = 0.0

            while True:
                frame = camera.read()
                if frame is not None:
                    stream.update(frame)

                now = time.monotonic()
                with lock:
                    payload = latest["payload"]
                    received_at = latest["received_at"]

                # Ignore stale laptop results; return to a neutral pose if the link is lost.
                fresh = payload is not None and now - received_at < 1.0
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
                        body_yaw=(None if body_yaw is None else __import__("math").radians(float(body_yaw))),
                    )
                else:
                    mini.set_target(head=create_head_pose(), antennas=list(NEUTRAL_ANTENNAS), body_yaw=0.0)

                if now - last_status >= 2.0:
                    if fresh:
                        print(f"Laptop inference connected; model={payload.get('model', '?')} FPS={payload.get('fps', 0):.1f}")
                    else:
                        print("Waiting for fresh laptop inference results...")
                    last_status = now
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.close()
        if camera is not None:
            camera.close()
        result_server.shutdown()
        result_server.server_close()


if __name__ == "__main__":
    main()