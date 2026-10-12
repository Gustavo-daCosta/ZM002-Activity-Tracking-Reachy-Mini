"""MoveNet SinglePose Lightning pose-estimation backends.

Backends:
- MoveNet ONNX: uses NVIDIA CUDA when available, with CPU fallback for
  operations that are not supported by CUDA.
- MoveNet LiteRT/TFLite: uses the CPU interpreter.

The ONNX backend is intended for GPU inference on the laptop.
The LiteRT backend remains available for the robot's existing CPU setup.
"""

import time

import cv2
import numpy as np

from core.pose_backends import PoseResult
from core.vision import download_model


INPUT_SIZE = 192

ONNX_FILE = "movenet-lightning.onnx"
ONNX_URL = (
    "https://huggingface.co/Xenova/movenet-singlepose-lightning/"
    "resolve/main/onnx/model.onnx"
)

TFLITE_FILE = "movenet-lightning-int8.tflite"
TFLITE_URL = (
    "https://huggingface.co/nxp/movenet-imx/"
    "resolve/main/original_model/movenet_quant.tflite"
)


def letterbox(frame_bgr, size):
    """Resize an image while preserving its aspect ratio, then pad it.

    Args:
        frame_bgr: Input image in BGR format.
        size: Width and height of the resulting square image.

    Returns:
        RGB image, resize scale, and (pad_x, pad_y).
    """
    height, width = frame_bgr.shape[:2]

    scale = size / max(height, width)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(
        frame_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2

    square = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    square[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width,
    ] = resized

    square_rgb = cv2.cvtColor(
        square,
        cv2.COLOR_BGR2RGB,
    )

    return square_rgb, scale, (pad_x, pad_y)


def decode_keypoints(output, scale, pad, frame_shape):
    """Convert MoveNet output into normalized COCO-17 keypoints.

    Returns:
        Float32 array with shape (17, 3): x, y, confidence.
    """
    height, width = frame_shape[:2]
    pad_x, pad_y = pad

    raw = np.asarray(output).reshape(-1, 3)

    keypoints = np.zeros(
        (raw.shape[0], 3),
        dtype=np.float32,
    )

    keypoints[:, 0] = (
        raw[:, 1] * INPUT_SIZE - pad_x
    ) / scale / width

    keypoints[:, 1] = (
        raw[:, 0] * INPUT_SIZE - pad_y
    ) / scale / height

    keypoints[:, 2] = raw[:, 2]

    return keypoints


class MoveNetBackend:
    """MoveNet FP32 ONNX using the NVIDIA CUDA Execution Provider."""

    name = "movenet-lightning"
    default_min_score = 0.3

    def __init__(self):
        """Load the ONNX model and initialize CUDA inference."""
        import onnxruntime as ort

        # Fail explicitly rather than silently running the entire model on CPU
        # when the installed ONNX Runtime has no CUDA provider.
        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "MoveNet GPU inference was requested, but "
                "CUDAExecutionProvider is unavailable.\n"
                f"Available providers: {available_providers}\n"
                "Install onnxruntime-gpu with compatible CUDA/cuDNN "
                "dependencies in the active Python environment."
            )

        # Load CUDA and cuDNN libraries installed through NVIDIA's Python
        # packages, when supported by this ONNX Runtime version.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")

        path = download_model(
            ONNX_FILE,
            ONNX_URL,
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Prioritize the NVIDIA GPU. ONNX Runtime can place unsupported
        # operations on the CPU through the second provider.
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                },
            ),
            "CPUExecutionProvider",
        ]

        self._session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=providers,
        )

        active_providers = self._session.get_providers()

        # Prevent silently accepting a session that failed to initialize CUDA.
        if "CUDAExecutionProvider" not in active_providers:
            self._session = None

            raise RuntimeError(
                "ONNX Runtime did not activate CUDA for MoveNet.\n"
                f"Session providers: {active_providers}\n"
                "Check CUDA/cuDNN library compatibility and installation."
            )

        self._input = self._session.get_inputs()[0].name

        print("MoveNet ONNX initialized.")
        print("Active session providers:", active_providers)
        print(
            "CUDA provider options:",
            self._session.get_provider_options().get(
                "CUDAExecutionProvider",
                {},
            ),
        )

    def infer(self, frame_bgr) -> PoseResult:
        """Run GPU-accelerated MoveNet inference on one BGR frame."""
        start = time.perf_counter()

        square, scale, pad = letterbox(
            frame_bgr,
            INPUT_SIZE,
        )

        output = self._session.run(
            None,
            {
                self._input: square[None].astype(np.int32),
            },
        )[0]

        return _result(
            output,
            scale,
            pad,
            frame_bgr.shape,
            start,
            self.default_min_score,
        )

    def close(self):
        """Release the ONNX Runtime session."""
        self._session = None


class MoveNetTFLiteBackend:
    """MoveNet INT8 using the LiteRT CPU interpreter."""

    name = "movenet-tflite"
    default_min_score = 0.3

    def __init__(self, threads=4):
        """Load the TFLite model using LiteRT."""
        from ai_edge_litert.interpreter import Interpreter

        path = download_model(
            TFLITE_FILE,
            TFLITE_URL,
        )

        self._interpreter = Interpreter(
            model_path=str(path),
            num_threads=threads,
        )

        self._interpreter.allocate_tensors()

        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

        print("MoveNet LiteRT initialized on CPU.")

    def infer(self, frame_bgr) -> PoseResult:
        """Run LiteRT inference on one BGR frame."""
        start = time.perf_counter()

        square, scale, pad = letterbox(
            frame_bgr,
            INPUT_SIZE,
        )

        self._interpreter.set_tensor(
            self._input["index"],
            square[None].astype(self._input["dtype"]),
        )

        self._interpreter.invoke()

        output = self._interpreter.get_tensor(
            self._output["index"],
        )

        return _result(
            output,
            scale,
            pad,
            frame_bgr.shape,
            start,
            self.default_min_score,
        )

    def close(self):
        """Release the LiteRT interpreter."""
        self._interpreter = None


def _result(output, scale, pad, frame_shape, start, min_score):
    """Decode model output and measure pose-inference time."""
    keypoints = decode_keypoints(
        output,
        scale,
        pad,
        frame_shape,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    if not (keypoints[:, 2] >= min_score).any():
        return PoseResult(
            timings={"pose": elapsed_ms},
        )

    return PoseResult(
        keypoints=keypoints,
        timings={"pose": elapsed_ms},
    )