"""ViTPose-S (COCO, ONNX): top-down pose — person detector first, then keypoints on the person crop.

Pre/post-processing follows easy_ViTPose (github.com/JunkyByte/easy_ViTPose): RGB, /255, ImageNet mean/std,
3:4 crop, heatmap argmax decoded with UDP coordinates.
"""

import time

import cv2
import numpy as np

from core.pose_backends import PoseResult

INPUT_W, INPUT_H = 192, 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def box_to_crop(box, padding=1.25):
    """Pixel box (x0, y0, x1, y1) -> crop (cx, cy, w, h) with margin and the model's 3:4 aspect ratio."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * padding, (y1 - y0) * padding
    aspect = INPUT_W / INPUT_H
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    return cx, cy, w, h


def crop_affine(cx, cy, w, h):
    """2x3 matrix for cv2.warpAffine: crop region -> INPUT_W x INPUT_H (outside the frame is padded black)."""
    sx, sy = INPUT_W / w, INPUT_H / h
    return np.array(
        [[sx, 0.0, -(cx - w / 2) * sx], [0.0, sy, -(cy - h / 2) * sy]],
        dtype=np.float32,
    )


def preprocess(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def heatmaps_to_keypoints(heatmaps, cx, cy, w, h, frame_w, frame_h):
    """(K, H, W) heatmaps -> (K, 3) normalized x, y in the frame and peak score."""
    num, hm_h, hm_w = heatmaps.shape
    x0, y0 = cx - w / 2, cy - h / 2
    keypoints = np.zeros((num, 3), np.float32)
    for j in range(num):
        hm = heatmaps[j]
        py, px = divmod(int(np.argmax(hm)), hm_w)
        fx, fy = float(px), float(py)
        # Quarter-pixel shift toward the higher neighbor reduces quantization error.
        if 0 < px < hm_w - 1:
            fx += 0.25 * np.sign(hm[py, px + 1] - hm[py, px - 1])
        if 0 < py < hm_h - 1:
            fy += 0.25 * np.sign(hm[py + 1, px] - hm[py - 1, px])
        x = x0 + fx * w / (hm_w - 1)
        y = y0 + fy * h / (hm_h - 1)
        keypoints[j] = (x / frame_w, y / frame_h, hm[py, px])
    return keypoints


def largest_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


DETECTOR_FILE = "efficientdet_lite0.tflite"
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)
POSE_FILE = "vitpose-s-coco.onnx"
POSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-s-coco.onnx"


class ViTPoseBackend:
    name = "vitpose-s"
    default_min_score = 0.3  # heatmap peaks are lower than BlazePose visibility

    def __init__(self):
        import mediapipe as mp
        import onnxruntime as ort

        from core.vision import download_model

        vision = mp.tasks.vision
        self._mp = mp
        self._detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(download_model(DETECTOR_FILE, DETECTOR_URL)),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.IMAGE,
                category_allowlist=["person"],
                score_threshold=0.4,
                max_results=3,
            )
        )
        # CPU only: comparable with the Raspberry Pi, and avoids CoreML differences on macOS.
        self._session = ort.InferenceSession(
            str(download_model(POSE_FILE, POSE_URL)), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def _detect_person(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        boxes = [
            (b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height)
            for b in (d.bounding_box for d in result.detections)
        ]
        return largest_box(boxes)

    def infer(self, frame_bgr) -> PoseResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        start = time.perf_counter()
        box = self._detect_person(frame_bgr)
        detected = time.perf_counter()
        timings = {"detector": (detected - start) * 1000}
        if box is None:
            return PoseResult(timings=timings)

        cx, cy, w, h = box_to_crop(box)
        crop = cv2.warpAffine(frame_bgr, crop_affine(cx, cy, w, h), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
        heatmaps = self._session.run(None, {self._input_name: preprocess(crop)})[0][0]
        keypoints = heatmaps_to_keypoints(heatmaps, cx, cy, w, h, frame_w, frame_h)
        timings["pose"] = (time.perf_counter() - detected) * 1000
        normalized_box = (box[0] / frame_w, box[1] / frame_h, box[2] / frame_w, box[3] / frame_h)
        return PoseResult(keypoints=keypoints, box=normalized_box, timings=timings)

    def close(self):
        self._detector.close()
