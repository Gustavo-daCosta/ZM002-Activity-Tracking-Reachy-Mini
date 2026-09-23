from types import SimpleNamespace

import numpy as np
import pytest

from core.pose_backends import BACKENDS, COCO_KEYPOINTS, PoseResult, create_backend
from core.pose_backends.blazepose import BLAZEPOSE_TO_COCO, landmarks_to_coco


def test_registry_names():
    assert list(BACKENDS) == ["blazepose-lite", "blazepose-full", "movenet-lightning", "movenet-tflite", "vitpose-s"]


def test_create_backend_unknown_name():
    with pytest.raises(ValueError, match="blazepose-lite"):
        create_backend("openpose")


def test_pose_result_defaults():
    result = PoseResult()
    assert result.keypoints is None and result.box is None and result.timings == {}


def test_blazepose_to_coco_mapping():
    assert len(COCO_KEYPOINTS) == 17
    assert BLAZEPOSE_TO_COCO == [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def test_landmarks_to_coco_picks_the_right_landmarks():
    landmarks = [SimpleNamespace(x=i / 100, y=i / 50, visibility=i / 33) for i in range(33)]
    coco = landmarks_to_coco(landmarks)
    assert coco.shape == (17, 3) and coco.dtype == np.float32
    left_shoulder = COCO_KEYPOINTS.index("left_shoulder")
    assert coco[left_shoulder] == pytest.approx([0.11, 0.22, 11 / 33])
    right_ankle = COCO_KEYPOINTS.index("right_ankle")
    assert coco[right_ankle] == pytest.approx([0.28, 0.56, 28 / 33])
