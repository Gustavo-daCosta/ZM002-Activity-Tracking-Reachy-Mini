"""Webcam access, pose model downloads and drawing helpers."""

import sys
import time
import urllib.request
from pathlib import Path

import cv2

MODELS_DIR = Path(__file__).parent / "models"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

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


def model_path(variant: str) -> Path:
    """Return the local BlazePose .task file, downloading it on first use."""
    return download_model(f"pose_landmarker_{variant}.task", MODEL_URLS[variant])


class PoseDetector:
    """MediaPipe PoseLandmarker in VIDEO mode (uses tracking between frames)."""

    def __init__(self, variant: str = "lite"):
        """Create the landmarker.

        Args:
            variant: "lite" or "full".
        """
        # Imported here: the mediapipe aarch64 binaries abort on the robot's CPU (no AES extensions).
        import mediapipe as mp

        self._mp = mp
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path(variant)),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,  # the GPU delegate crashes on macOS
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._start = time.monotonic()
        self._last_ts = -1

    def detect(self, frame_bgr):
        """Return the 33 landmarks of the first person, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int((time.monotonic() - self._start) * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def close(self):
        """Release the landmarker."""
        self._landmarker.close()


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
