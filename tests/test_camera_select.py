import argparse

import pytest

from core.vision import DEFAULT_CAMERA, add_camera_argument, resolve_camera

CAMERAS = [(0, "Câmera do Caio"), (1, "Câmera FaceTime HD"), (2, "Câmera da Visualização da Mesa Caio")]


def test_numeric_spec_is_an_index():
    assert resolve_camera("1", CAMERAS) == 1
    assert resolve_camera("7", CAMERAS) == 7  # allowed even if not listed


def test_name_substring_is_case_insensitive():
    assert resolve_camera("facetime", CAMERAS) == 1
    assert resolve_camera("FaceTime HD", CAMERAS) == 1


def test_unknown_name_lists_cameras():
    with pytest.raises(ValueError, match="Câmera FaceTime HD"):
        resolve_camera("logitech", CAMERAS)


def test_ambiguous_name_lists_matches():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_camera("caio", CAMERAS)


def test_camera_argument_defaults_to_mac_camera():
    parser = argparse.ArgumentParser()
    add_camera_argument(parser)
    assert parser.parse_args([]).camera == DEFAULT_CAMERA == "FaceTime"
    assert parser.parse_args(["--camera", "0"]).camera == "0"
