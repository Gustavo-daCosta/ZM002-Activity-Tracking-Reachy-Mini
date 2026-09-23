"""Interchangeable pose estimation backends returning COCO-17 keypoints.

Heavy libraries (MediaPipe, ONNX Runtime) are imported only when a backend is created.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol, Tuple

import numpy as np

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 0, 5, 6, 11, 12
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 7, 8, 9, 10
LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 13, 14, 15, 16


@dataclass
class PoseResult:
    """keypoints: (17, 3) float32 with x, y normalized to the frame and a score; None if no person."""

    keypoints: Optional[np.ndarray] = None
    box: Optional[Tuple[float, float, float, float]] = None  # normalized x0, y0, x1, y1
    timings: Dict[str, float] = field(default_factory=dict)  # stage -> milliseconds


class PoseBackend(Protocol):
    name: str
    default_min_score: float

    def infer(self, frame_bgr: np.ndarray) -> PoseResult: ...

    def close(self) -> None: ...


def _blazepose(variant: str) -> Callable[[], PoseBackend]:
    def factory():
        from core.pose_backends.blazepose import BlazePoseBackend

        return BlazePoseBackend(variant)

    return factory


def _movenet() -> PoseBackend:
    from core.pose_backends.movenet import MoveNetBackend

    return MoveNetBackend()


def _movenet_tflite() -> PoseBackend:
    from core.pose_backends.movenet_tflite import MoveNetTFLiteBackend

    return MoveNetTFLiteBackend()


def _vitpose() -> PoseBackend:
    from core.pose_backends.vitpose import ViTPoseBackend

    return ViTPoseBackend()


BACKENDS: Dict[str, Callable[[], PoseBackend]] = {
    "blazepose-lite": _blazepose("lite"),
    "blazepose-full": _blazepose("full"),
    "movenet-lightning": _movenet,
    "movenet-tflite": _movenet_tflite,
    "vitpose-s": _vitpose,
}


def create_backend(name: str) -> PoseBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown pose model {name!r}; choose one of: {', '.join(BACKENDS)}")
    return BACKENDS[name]()
