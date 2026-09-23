import numpy as np
import pytest

from core.pose_backends.vitpose import (
    INPUT_H, INPUT_W, box_to_crop, crop_affine, heatmaps_to_keypoints, largest_box, preprocess,
)


def test_box_to_crop_tall_box_grows_width():
    cx, cy, w, h = box_to_crop((100, 0, 200, 400), padding=1.25)
    assert (cx, cy) == (150, 200)
    assert h == pytest.approx(500)
    assert w == pytest.approx(375)  # 500 * 192 / 256


def test_box_to_crop_wide_box_grows_height():
    cx, cy, w, h = box_to_crop((0, 0, 400, 100), padding=1.0)
    assert w == pytest.approx(400)
    assert h == pytest.approx(400 * 256 / 192)


def test_crop_affine_maps_region_corners():
    m = crop_affine(300, 200, 150, 200)
    top_left = m @ np.array([225, 100, 1])
    bottom_right = m @ np.array([375, 300, 1])
    assert top_left == pytest.approx([0, 0], abs=1e-3)
    assert bottom_right == pytest.approx([INPUT_W, INPUT_H], abs=1e-3)


def test_preprocess_shape_color_order_and_normalization():
    crop = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
    crop[..., 0] = 255  # pure blue in BGR
    x = preprocess(crop)
    assert x.shape == (1, 3, INPUT_H, INPUT_W) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((0 - 0.485) / 0.229)   # R
    assert x[0, 2, 0, 0] == pytest.approx((1 - 0.406) / 0.225)   # B


def test_heatmap_peak_maps_back_to_frame():
    heatmaps = np.zeros((17, 64, 48), np.float32)
    heatmaps[3, 32, 24] = 0.9
    kps = heatmaps_to_keypoints(heatmaps, cx=320, cy=240, w=300, h=400, frame_w=640, frame_h=480)
    assert kps.shape == (17, 3)
    assert kps[3, 0] == pytest.approx((170 + 24 * 300 / 47) / 640)
    assert kps[3, 1] == pytest.approx((40 + 32 * 400 / 63) / 480)
    assert kps[3, 2] == pytest.approx(0.9)


def test_heatmap_quarter_pixel_shift_toward_higher_neighbor():
    heatmaps = np.zeros((1, 64, 48), np.float32)
    heatmaps[0, 10, 20] = 1.0
    heatmaps[0, 10, 21] = 0.5
    heatmaps[0, 9, 20] = 0.3
    kps = heatmaps_to_keypoints(heatmaps, cx=47 / 2, cy=63 / 2, w=47, h=63, frame_w=47, frame_h=63)
    assert kps[0, 0] * 47 == pytest.approx(20.25)
    assert kps[0, 1] * 63 == pytest.approx(9.75)


def test_largest_box():
    assert largest_box([]) is None
    assert largest_box([(0, 0, 10, 10), (0, 0, 50, 20), (5, 5, 20, 20)]) == (0, 0, 50, 20)
