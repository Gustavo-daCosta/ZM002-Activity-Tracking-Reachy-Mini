"""BlazePose (MediaPipe Pose Landmarker): single-stage, 33 landmarks mapped to COCO-17."""

import time

import numpy as np

from core.pose_backends import PoseResult

# COCO-17 order -> BlazePose landmark index.
BLAZEPOSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def landmarks_to_coco(landmarks) -> np.ndarray:
    """33 MediaPipe landmarks -> (17, 3) array of normalized x, y and visibility as score."""
    return np.array(
        [[landmarks[i].x, landmarks[i].y, landmarks[i].visibility] for i in BLAZEPOSE_TO_COCO],
        dtype=np.float32,
    )


class BlazePoseBackend:
    """Pose backend on MediaPipe's landmarker (Mac only: mediapipe aborts on the robot's CPU)."""

    default_min_score = 0.5

    def __init__(self, variant: str):
        """Create the landmarker for "lite" or "full"."""
        from core.vision import PoseDetector

        self.name = f"blazepose-{variant}"
        self._detector = PoseDetector(variant)

    def infer(self, frame_bgr) -> PoseResult:
        """Run the landmarker on one BGR frame."""
        start = time.perf_counter()
        landmarks = self._detector.detect(frame_bgr)
        elapsed_ms = (time.perf_counter() - start) * 1000
        keypoints = landmarks_to_coco(landmarks) if landmarks else None
        return PoseResult(keypoints=keypoints, timings={"pose": elapsed_ms})

    def close(self):
        """Release the landmarker."""
        self._detector.close()
