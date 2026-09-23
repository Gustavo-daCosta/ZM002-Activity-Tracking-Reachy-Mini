"""MoveNet Lightning int8 on TFLite/XNNPACK: the fast path on the robot.

onnxruntime on the robot only has the plain CPU provider (~95 ms per frame for MoveNet fp32). The same model
quantized to int8 under LiteRT's XNNPACK delegate runs in ~34 ms on the Cortex-A72, and its keypoints stay
within ~2 % of the frame width of the fp32 ones - well inside the tolerance of shoulder-normalized features.

Model: huggingface.co/nxp/movenet-imx (original_model/movenet_quant.tflite), uint8 [1,192,192,3] in,
float [1,1,17,3] out (y, x, score) - same layout as the ONNX export, so the decoding is shared.
"""

import time

import numpy as np

from core.pose_backends import PoseResult
from core.pose_backends.movenet import INPUT_SIZE, decode_keypoints, letterbox
from core.vision import download_model

MODEL_FILE = "movenet-lightning-int8.tflite"
MODEL_URL = "https://huggingface.co/nxp/movenet-imx/resolve/main/original_model/movenet_quant.tflite"


class MoveNetTFLiteBackend:
    name = "movenet-tflite"
    default_min_score = 0.3

    def __init__(self, threads=4):
        from ai_edge_litert.interpreter import Interpreter

        path = download_model(MODEL_FILE, MODEL_URL)
        self._interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def infer(self, frame_bgr) -> PoseResult:
        start = time.perf_counter()
        square, scale, pad = letterbox(frame_bgr, INPUT_SIZE)
        self._interpreter.set_tensor(self._input["index"], square[None].astype(self._input["dtype"]))
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output["index"])
        keypoints = decode_keypoints(output, scale, pad, frame_bgr.shape)
        ms = (time.perf_counter() - start) * 1000
        if not (keypoints[:, 2] >= self.default_min_score).any():
            return PoseResult(timings={"pose": ms})
        return PoseResult(keypoints=keypoints, timings={"pose": ms})

    def close(self):
        self._interpreter = None
