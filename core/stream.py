"""MJPEG preview server: watch the robot camera with the model overlay from a browser on the network."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"
PAGE = b"""<!doctype html><html><head><title>Reachy Mini camera</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream.mjpg"></body></html>"""


class MjpegServer:
    """Holds the latest JPEG; every connected client gets it as fast as it can consume it."""

    def __init__(self, port=8080, host="0.0.0.0", quality=70):
        """Bind the server (not started yet) on `host:port` (port 0 = any free port)."""
        self._jpeg = None
        self._condition = threading.Condition()
        self._quality = quality
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="mjpeg", daemon=True)

    @property
    def port(self):
        """The bound port."""
        return self._server.server_address[1]

    def start(self):
        """Start serving and return self."""
        self._thread.start()
        return self

    def close(self):
        """Stop serving and release the socket."""
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

        def log_message(self, *args):
            pass  # keep stdout for the app's own log

    return Handler
