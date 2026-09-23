"""MJPEG preview server: the robot annotates frames, the Mac watches them in a browser."""

import urllib.request

import numpy as np
import pytest

from core.stream import MjpegServer


def read_some(url, size=512):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.headers["Content-Type"], response.read(size)


def test_serves_the_pushed_frame_as_multipart_jpeg():
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)  # >512 bytes of JPEG
    with MjpegServer(port=0) as server:
        server.update(frame)
        content_type, payload = read_some(server.url)

    assert "multipart/x-mixed-replace" in content_type
    assert b"\xff\xd8\xff" in payload  # JPEG magic


def test_index_page_links_the_stream():
    with MjpegServer(port=0) as server:
        with urllib.request.urlopen(server.url.replace("/stream.mjpg", "/"), timeout=5) as response:
            page = response.read()

    assert b"stream.mjpg" in page


def test_update_before_any_client_does_not_fail():
    with MjpegServer(port=0) as server:
        server.update(np.zeros((8, 8, 3), np.uint8))
        server.update(np.zeros((8, 8, 3), np.uint8))


def test_url_uses_the_bound_port():
    with MjpegServer(port=0) as server:
        assert server.port != 0
        assert str(server.port) in server.url


def test_closed_server_stops_serving():
    server = MjpegServer(port=0)
    server.start()
    url = server.url
    server.close()

    with pytest.raises(Exception):
        read_some(url)
