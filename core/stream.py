"""MJPEG preview server: watch what the robot's camera sees, with the model overlay, from another machine.

The robot has no display, and sending raw frames back would need a second video pipeline. Instead the robot
annotates the frame it just processed and serves it as multipart JPEG, so any browser on the same network can
open http://<robot>:<port>/ and watch. Encoding a 640x360 frame costs a few milliseconds.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"
PAGE = b"""<!doctype html><html><head><title>Reachy Mini camera</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream.mjpg"></body></html>"""


class MjpegServer:
    """Holds the latest frame; every connected client gets it as fast as it can consume it."""

    def __init__(self, port=8080, host="0.0.0.0", quality=70, fps=15.0):
        self._jpeg = None
        self._condition = threading.Condition()
        self._quality = quality
        self._interval = 1.0 / fps
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="mjpeg", daemon=True)

    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def url(self):
        return f"http://localhost:{self.port}/stream.mjpg"

    def start(self):
        self._thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def update(self, frame):
        """Publish a frame (encoded once, whatever the number of clients)."""
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        with self._condition:
            self._jpeg = buffer.tobytes()
            self._condition.notify_all()

    def _wait_for_frame(self, last):
        with self._condition:
            if self._jpeg is last:
                self._condition.wait(timeout=1.0)
            return self._jpeg


def _make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html", PAGE)
            elif self.path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, content_type, body):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            last = None
            try:
                while True:
                    jpeg = server._wait_for_frame(last)
                    if jpeg is None or jpeg is last:
                        continue
                    last = jpeg
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):  # keep the robot's stdout for the wave log
            pass

    return Handler
