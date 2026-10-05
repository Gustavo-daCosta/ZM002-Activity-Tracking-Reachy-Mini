"""Cameras (webcam by name, robot camera through the SDK), model downloads and drawing helpers."""

import sys
import urllib.request
from pathlib import Path

import cv2

MODELS_DIR = Path(__file__).parent / "models"
# Cameras are selected by name: on macOS the OpenCV indices change when the iPhone (Continuity Camera)
# connects, but "FaceTime" always matches the built-in camera.
DEFAULT_CAMERA = "FaceTime"


def add_camera_argument(parser):
    """Add the `--camera` option to an argparse parser."""
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA,
        help=f"camera name substring or OpenCV index (default {DEFAULT_CAMERA!r}; see camera_check --list)",
    )


def list_cameras() -> list:
    """(OpenCV index, name) for every camera, in OpenCV's index order."""
    from cv2_enumerate_cameras import enumerate_cameras

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    return [(camera.index, camera.name) for camera in enumerate_cameras(backend)]


def resolve_camera(spec: str, cameras=None) -> int:
    """Map a camera spec to an OpenCV index.

    Args:
        spec: A numeric index, or a case-insensitive substring of the camera name.
        cameras: (index, name) pairs; None lists them.

    Returns:
        The OpenCV index.

    Raises:
        ValueError: No camera matches, or several do.
    """
    if spec.isdigit():
        return int(spec)
    cameras = list_cameras() if cameras is None else cameras
    available = ", ".join(f"{index}: {name}" for index, name in cameras) or "none"
    matches = [(index, name) for index, name in cameras if spec.lower() in name.lower()]
    if not matches:
        raise ValueError(f"No camera matching {spec!r}. Available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Camera name {spec!r} is ambiguous: {matches}. Available: {available}")
    return matches[0][0]


def open_camera(spec, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Open a webcam by spec (see `resolve_camera`) at the requested size.

    Raises:
        RuntimeError: The camera could not be opened.
    """
    index = resolve_camera(str(spec))
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {spec!r} (index {index}). List cameras with "
            "`python -m sim.camera_check --list` and check macOS camera permissions."
        )
    print(f"Using camera {index} ({spec!r})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def download_model(filename: str, url: str) -> Path:
    """Return models/<filename>, downloading it atomically on first use.

    Raises:
        SystemExit: The download failed (with manual instructions).
    """
    path = MODELS_DIR / filename
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, partial)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download {filename}: {exc}\nDownload it manually from\n  {url}\nand save it as\n  {path}"
        ) from exc
    partial.rename(path)
    return path


# COCO-17 skeleton (see core.pose_backends.COCO_KEYPOINTS for the index order).
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_coco_skeleton(frame, keypoints, min_score):
    """Draw confident COCO keypoints (normalized x, y, score) and the bones between them."""
    h, w = frame.shape[:2]
    confident = keypoints[:, 2] >= min_score
    points = [(int(x * w), int(y * h)) for x, y, _ in keypoints]
    for a, b in COCO_SKELETON:
        if confident[a] and confident[b]:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for i, point in enumerate(points):
        if confident[i]:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)


def put_text(frame, text, row, color=(255, 255, 255)):
    """Draw one line of overlay text at row `row` (1-based)."""
    cv2.putText(frame, text, (10, 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class RobotCameraSource:
    """Direct V4L2/OpenCV access to the Reachy Mini camera, bypassing the 10 FPS SDK media feed."""

    def __init__(self, device="/dev/video0", width=1920, height=1080, fps=60):
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open Reachy Mini camera at {device}. "
                "Run `v4l2-ctl --list-devices` to find the correct device."
            )

        # The Reachy Mini UVC camera exposes high-FPS modes through MJPEG/V4L2.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(
            f"Direct camera: {device} {actual_width}x{actual_height} "
            f"@ {actual_fps:.1f} FPS"
        )

    def read(self):
        """Read the next frame directly from the physical camera."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        """Release the physical camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def downscale(frame, width):
    """Shrink a frame to `width` keeping its aspect ratio (never upscales)."""
    if width <= 0 or frame.shape[1] <= width:
        return frame
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
