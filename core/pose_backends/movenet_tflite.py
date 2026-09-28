"""MoveNet Lightning int8 on TFLite/XNNPACK: the fast path on the robot (~34 ms vs ~95 ms on onnxruntime).

Model: huggingface.co/nxp/movenet-imx (movenet_quant.tflite), uint8 [1,192,192,3] in, float [1,1,17,3]
out, same layout as the ONNX export so the decoding is shared.
"""

import time

from core.pose_backends import PoseResult
from core.pose_backends.movenet import INPUT_SIZE, decode_keypoints, letterbox
from core.vision import download_model

MODEL_FILE = "movenet-lightning-int8.tflite"
MODEL_URL = "https://huggingface.co/nxp/movenet-imx/resolve/main/original_model/movenet_quant.tflite"


class MoveNetTFLiteBackend:
    """MoveNet int8 on the LiteRT interpreter."""

    name = "movenet-tflite"
    default_min_score = 0.3

    def __init__(self, threads=4):
        """Load the interpreter, downloading the model on first use."""
        from ai_edge_litert.interpreter import Interpreter

        path = download_model(MODEL_FILE, MODEL_URL)
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
        keypoints = decode_keypoints(output, scale, pad, frame_bgr.shape)
        ms = (time.perf_counter() - start) * 1000
        if not (keypoints[:, 2] >= self.default_min_score).any():
            return PoseResult(timings={"pose": ms})
        return PoseResult(keypoints=keypoints, timings={"pose": ms})

    def close(self):
        """Drop the interpreter."""
        self._interpreter = None
