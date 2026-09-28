"""Interchangeable pose estimation backends returning COCO-17 keypoints.

Heavy libraries (MediaPipe, ONNX Runtime, LiteRT) are imported only when a backend is created.
"""

import importlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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
    """Output of one inference.

    Attributes:
        keypoints: (17, 3) float32 with x, y normalized to the frame and a score; None if no person.
        box: Normalized (x0, y0, x1, y1) of the person, when the backend has one.
        timings: Milliseconds per stage.
    """

    keypoints: Optional[np.ndarray] = None
    box: Optional[Tuple[float, float, float, float]] = None
    timings: Dict[str, float] = field(default_factory=dict)


# name -> (module, class, constructor args)
BACKENDS = {
    "blazepose-lite": ("core.pose_backends.blazepose", "BlazePoseBackend", ("lite",)),
    "blazepose-full": ("core.pose_backends.blazepose", "BlazePoseBackend", ("full",)),
    "movenet-lightning": ("core.pose_backends.movenet", "MoveNetBackend", ()),
    "movenet-tflite": ("core.pose_backends.movenet_tflite", "MoveNetTFLiteBackend", ()),
    "vitpose-s": ("core.pose_backends.vitpose", "ViTPoseBackend", ()),
}


def create_backend(name: str):
    """Instantiate the backend registered under `name`.

    Args:
        name: A key of `BACKENDS`.

    Returns:
        An object with `name`, `default_min_score`, `infer(frame_bgr)` and `close()`.

    Raises:
        ValueError: Unknown name.
    """
    if name not in BACKENDS:
        raise ValueError(f"Unknown pose model {name!r}; choose one of: {', '.join(BACKENDS)}")
    module, cls, args = BACKENDS[name]
    return getattr(importlib.import_module(module), cls)(*args)
