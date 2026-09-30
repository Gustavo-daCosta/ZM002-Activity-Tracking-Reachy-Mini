"""MoveNet SinglePose Lightning (COCO-17): single-shot, no person detector.

Two runtimes share the pre/post-processing: fp32 ONNX on onnxruntime (~95 ms on the robot) and int8
TFLite on LiteRT/XNNPACK (~34 ms, the robot default). Input [1,192,192,3] RGB letterboxed with black
padding, output float [1,1,17,3] as (y, x, score) in padded-square space.
"""

import time

import cv2
import numpy as np

from core.pose_backends import PoseResult
from core.vision import download_model

INPUT_SIZE = 192
ONNX_FILE = "movenet-lightning.onnx"
ONNX_URL = "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"
TFLITE_FILE = "movenet-lightning-int8.tflite"
TFLITE_URL = "https://huggingface.co/nxp/movenet-imx/resolve/main/original_model/movenet_quant.tflite"


def letterbox(frame_bgr, size):
    """Resize keeping the aspect ratio and pad to size x size.

    Returns:
        (square RGB image, scale, (pad_x, pad_y)).
    """
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
    keypoints[:, 0] = (raw[:, 1] * INPUT_SIZE - pad_x) / scale / w
    keypoints[:, 1] = (raw[:, 0] * INPUT_SIZE - pad_y) / scale / h
    keypoints[:, 2] = raw[:, 2]
    return keypoints


class MoveNetBackend:
    """MoveNet fp32 on onnxruntime (CPU provider only)."""

    name = "movenet-lightning"
    default_min_score = 0.3

    def __init__(self):
        """Load the ONNX session, downloading the model on first use."""
        import onnxruntime as ort

        path = download_model(ONNX_FILE, ONNX_URL)
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def infer(self, frame_bgr) -> PoseResult:
        """Run the model on one BGR frame."""
        start = time.perf_counter()
        square, scale, pad = letterbox(frame_bgr, INPUT_SIZE)
        output = self._session.run(None, {self._input: square[None].astype(np.int32)})[0]
        return _result(output, scale, pad, frame_bgr.shape, start, self.default_min_score)

    def close(self):
        """Drop the session."""
        self._session = None


class MoveNetTFLiteBackend:
    """MoveNet int8 on the LiteRT interpreter."""

    name = "movenet-tflite"
    default_min_score = 0.3

    def __init__(self, threads=4):
        """Load the interpreter, downloading the model on first use."""
        from ai_edge_litert.interpreter import Interpreter

        path = download_model(TFLITE_FILE, TFLITE_URL)
        self._interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def infer(self, frame_bgr) -> PoseResult:
        """Run the model on one BGR frame."""
        start = time.perf_counter()
        square, scale, pad = letterbox(frame_bgr, INPUT_SIZE)
        self._interpreter.set_tensor(self._input["index"], square[None].astype(self._input["dtype"]))
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output["index"])
        return _result(output, scale, pad, frame_bgr.shape, start, self.default_min_score)

    def close(self):
        """Drop the interpreter."""
        self._interpreter = None


def _result(output, scale, pad, frame_shape, start, min_score):
    """Decode one model output; no keypoints when no joint clears `min_score`."""
    keypoints = decode_keypoints(output, scale, pad, frame_shape)
    ms = (time.perf_counter() - start) * 1000
    if not (keypoints[:, 2] >= min_score).any():
        return PoseResult(timings={"pose": ms})
    return PoseResult(keypoints=keypoints, timings={"pose": ms})
