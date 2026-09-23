"""MoveNet letterboxing and keypoint decoding (pure functions, no model file needed)."""

import numpy as np
import pytest

from core.pose_backends.movenet import INPUT_SIZE, decode_keypoints, letterbox


def test_letterbox_pads_to_a_square_without_distortion():
    frame = np.zeros((360, 640, 3), np.uint8)
    square, scale, pad = letterbox(frame, INPUT_SIZE)

    assert square.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert scale == pytest.approx(INPUT_SIZE / 640)
    assert pad == (0, (INPUT_SIZE - round(360 * INPUT_SIZE / 640)) // 2)  # black bars top and bottom


def test_letterbox_pads_the_sides_for_a_tall_frame():
    frame = np.zeros((640, 360, 3), np.uint8)
    _, scale, pad = letterbox(frame, INPUT_SIZE)

    assert scale == pytest.approx(INPUT_SIZE / 640)
    assert pad[1] == 0 and pad[0] > 0


def test_decode_maps_a_center_keypoint_back_to_the_frame_center():
    frame_shape = (360, 640)
    _, scale, pad = letterbox(np.zeros((*frame_shape, 3), np.uint8), INPUT_SIZE)
    output = np.zeros((1, 1, 17, 3), np.float32)
    output[0, 0, :, :2] = 0.5  # y, x at the middle of the padded square
    output[0, 0, :, 2] = 0.9

    keypoints = decode_keypoints(output, scale, pad, frame_shape)

    assert keypoints.shape == (17, 3)
    assert keypoints[0, 0] == pytest.approx(0.5, abs=1e-3)
    assert keypoints[0, 1] == pytest.approx(0.5, abs=1e-3)
    assert keypoints[0, 2] == pytest.approx(0.9)


def test_decode_keeps_scores_and_returns_normalized_coordinates():
    frame_shape = (480, 640)
    _, scale, pad = letterbox(np.zeros((*frame_shape, 3), np.uint8), INPUT_SIZE)
    output = np.zeros((1, 1, 17, 3), np.float32)
    output[0, 0, 5] = (0.25, 0.75, 0.4)  # y, x, score of the left shoulder

    keypoints = decode_keypoints(output, scale, pad, frame_shape)

    assert 0.0 <= keypoints[5, 0] <= 1.0 and 0.0 <= keypoints[5, 1] <= 1.0
    assert keypoints[5, 1] < 0.5  # the padded top maps above the frame center
    assert keypoints[5, 2] == pytest.approx(0.4)
