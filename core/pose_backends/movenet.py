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

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER, PoseResult
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


# Crop tracking, as in Google's MoveNet demo: each frame is cut around the previous pose so the person
# fills the 192 px input. Letterboxing the robot's 16:9 wide-angle frame left ~192x108 px of image and the
# person a few dozen pixels tall (person detected in 52% of frames on the robot).
# Half the crop side is at least TORSO_MARGIN x the farthest torso joint from the body centre (the demo uses
# 1.9; arms raised straight up need more) and BODY_MARGIN x the farthest confident joint.
TORSO_MARGIN = 2.2
BODY_MARGIN = 1.3
MIN_CROP = 0.25  # smallest crop side, as a fraction of the frame's long side
TORSO = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


def crop_region(previous, min_score, frame_shape):
    """(x0, y0, side) in pixels: a square around the `previous` pose, or the whole frame letterboxed.

    The square may stick out of the frame; `crop_square` fills that part with black.
    """
    height, width = frame_shape[:2]
    long_side = max(width, height)
    whole = ((width - long_side) / 2, (height - long_side) / 2, long_side)
    if previous is None:
        return whole
    xy = previous[:, :2] * (width, height)
    confident = previous[:, 2] >= min_score
    if confident[[LEFT_HIP, RIGHT_HIP]].all():
        centre = xy[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
    elif confident[[LEFT_SHOULDER, RIGHT_SHOULDER]].all():
        centre = xy[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
    else:
        return whole
    reach = np.abs(xy - centre).max(axis=1)  # per joint, along the farther axis: the crop is square
    torso = [joint for joint in TORSO if confident[joint]]
    half = max(TORSO_MARGIN * reach[torso].max(), BODY_MARGIN * reach[confident].max(), MIN_CROP * long_side / 2)
    if 2 * half >= long_side:
        return whole
    return centre[0] - half, centre[1] - half, 2 * half


def crop_square(frame_bgr, region, size=INPUT_SIZE):
    """The `region` of the frame as a `size` x `size` RGB image, black where it leaves the frame."""
    x0, y0, side = region
    scale = size / side
    matrix = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    square = cv2.warpAffine(frame_bgr, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return cv2.cvtColor(square, cv2.COLOR_BGR2RGB)


def decode_keypoints(output, region, frame_shape):
    """Convert MoveNet output (relative to the crop) into frame-normalized COCO-17 keypoints.

    Returns:
        Float32 array with shape (17, 3): x, y, confidence.
    """
    height, width = frame_shape[:2]
    x0, y0, side = region
    raw = np.asarray(output).reshape(-1, 3)
    keypoints = np.zeros((raw.shape[0], 3), dtype=np.float32)
    keypoints[:, 0] = (x0 + raw[:, 1] * side) / width
    keypoints[:, 1] = (y0 + raw[:, 0] * side) / height
    keypoints[:, 2] = raw[:, 2]
    return keypoints


class MoveNetBackend:
    """MoveNet FP32 ONNX using the NVIDIA CUDA Execution Provider."""

    name = "movenet-lightning"
    default_min_score = 0.3

    def __init__(self):
        """Load the ONNX model and initialize CUDA inference."""
        self._previous = None  # last pose, where the next crop is centred
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

        region = crop_region(self._previous, self.default_min_score, frame_bgr.shape)
        square = crop_square(frame_bgr, region)

        output = self._session.run(
            None,
            {
                self._input: square[None].astype(np.int32),
            },
        )[0]

        result = _result(output, region, frame_bgr.shape, start, self.default_min_score)
        self._previous = result.keypoints
        return result

    def close(self):
        """Release the ONNX Runtime session."""
        self._session = None


class MoveNetTFLiteBackend:
    """MoveNet INT8 using the LiteRT CPU interpreter."""

    name = "movenet-tflite"
    default_min_score = 0.3

    def __init__(self, threads=4):
        """Load the TFLite model using LiteRT."""
        self._previous = None  # last pose, where the next crop is centred
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

        region = crop_region(self._previous, self.default_min_score, frame_bgr.shape)
        square = crop_square(frame_bgr, region)

        self._interpreter.set_tensor(
            self._input["index"],
            square[None].astype(self._input["dtype"]),
        )

        self._interpreter.invoke()

        output = self._interpreter.get_tensor(
            self._output["index"],
        )

        result = _result(output, region, frame_bgr.shape, start, self.default_min_score)
        self._previous = result.keypoints
        return result

    def close(self):
        """Release the LiteRT interpreter."""
        self._interpreter = None


def _result(output, region, frame_shape, start, min_score):
    """Decode model output and measure pose-inference time."""
    keypoints = decode_keypoints(output, region, frame_shape)

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