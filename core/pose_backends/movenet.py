"""MoveNet SinglePose Lightning (COCO-17, ONNX Runtime): single-shot, no person detector.

Chosen for the robot: the Raspberry Pi CM4 (Cortex-A72) has no AES/SHA extensions, and the mediapipe 1.x
aarch64 binaries abort with "compiled with aes enabled, but this feature is not available on this processor",
so BlazePose cannot run there. MoveNet is a single 192x192 model whose 17 keypoints already are COCO order.

Model: huggingface.co/Xenova/movenet-singlepose-lightning (onnx/model.onnx), input int32 [1,192,192,3] (RGB
0..255, aspect preserved with black padding), output float [1,1,17,3] as (y, x, score) in padded-square space.
"""

import time

import cv2
import numpy as np

from core.pose_backends import PoseResult
from core.vision import download_model

INPUT_SIZE = 192
MODEL_FILE = "movenet-lightning.onnx"
MODEL_URL = "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"


def letterbox(frame_bgr, size):
    """Resize keeping the aspect ratio and pad to size x size. Returns (square RGB, scale, (pad_x, pad_y))."""
    h, w = frame_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    square = np.zeros((size, size, 3), np.uint8)
    square[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return cv2.cvtColor(square, cv2.COLOR_BGR2RGB), scale, (pad_x, pad_y)


def decode_keypoints(output, scale, pad, frame_shape):
    """(1, 1, 17, 3) model output -> (17, 3) float32 with x, y normalized to the frame and the score."""
    h, w = frame_shape[:2]
    pad_x, pad_y = pad
    raw = np.asarray(output).reshape(-1, 3)
    keypoints = np.zeros((raw.shape[0], 3), np.float32)
    # Model coordinates are relative to the padded square: undo the padding and the resize.
    keypoints[:, 0] = (raw[:, 1] * INPUT_SIZE - pad_x) / scale / w
    keypoints[:, 1] = (raw[:, 0] * INPUT_SIZE - pad_y) / scale / h
    keypoints[:, 2] = raw[:, 2]
    return keypoints


class MoveNetBackend:
    name = "movenet-lightning"
    default_min_score = 0.3

    def __init__(self):
        import onnxruntime as ort

        path = download_model(MODEL_FILE, MODEL_URL)
        # CPU only: the robot has no GPU and this keeps the timings comparable with the other backends.
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def infer(self, frame_bgr) -> PoseResult:
        start = time.perf_counter()
        square, scale, pad = letterbox(frame_bgr, INPUT_SIZE)
        output = self._session.run(None, {self._input: square[None].astype(np.int32)})[0]
        keypoints = decode_keypoints(output, scale, pad, frame_bgr.shape)
        ms = (time.perf_counter() - start) * 1000
        if not (keypoints[:, 2] >= self.default_min_score).any():
            return PoseResult(timings={"pose": ms})
        return PoseResult(keypoints=keypoints, timings={"pose": ms})

    def close(self):
        self._session = None
