"""BlazePose (MediaPipe Pose Landmarker): single-stage, 33 landmarks mapped to COCO-17."""

import time

import cv2
import numpy as np

from core.pose_backends import PoseResult
from core.vision import download_model

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

# COCO-17 order -> BlazePose landmark index.
BLAZEPOSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def landmarks_to_coco(landmarks) -> np.ndarray:
    """33 MediaPipe landmarks -> (17, 3) array of normalized x, y and visibility as score."""
    return np.array(
        [[landmarks[i].x, landmarks[i].y, landmarks[i].visibility] for i in BLAZEPOSE_TO_COCO],
        dtype=np.float32,
    )


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
                model_asset_path=str(download_model(f"pose_landmarker_{variant}.task", MODEL_URLS[variant])),
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


class BlazePoseBackend:
    """Pose backend on MediaPipe's landmarker (Mac only: mediapipe aborts on the robot's CPU)."""

    default_min_score = 0.5

    def __init__(self, variant: str):
        """Create the landmarker for "lite" or "full"."""
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
