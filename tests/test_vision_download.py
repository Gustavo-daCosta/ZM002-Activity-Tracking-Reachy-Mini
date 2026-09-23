import numpy as np
import pytest

from core import vision


def test_download_model_fetches_and_caches(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"model-bytes")
    models = tmp_path / "models"
    monkeypatch.setattr(vision, "MODELS_DIR", models)

    path = vision.download_model("m.bin", source.as_uri())
    assert path == models / "m.bin"
    assert path.read_bytes() == b"model-bytes"

    source.unlink()  # second call must not download again
    assert vision.download_model("m.bin", source.as_uri()) == path


def test_download_model_failure_leaves_no_file(tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setattr(vision, "MODELS_DIR", models)
    missing = (tmp_path / "missing.bin").as_uri()

    with pytest.raises(SystemExit) as exc:
        vision.download_model("m.bin", missing)
    assert missing in str(exc.value)
    assert list(models.iterdir()) == []


def test_draw_coco_skeleton_skips_low_score_points():
    frame = np.zeros((100, 100, 3), np.uint8)
    keypoints = np.zeros((17, 3), np.float32)
    keypoints[5] = (0.2, 0.5, 0.9)   # left shoulder, confident
    keypoints[6] = (0.8, 0.5, 0.9)   # right shoulder, confident
    keypoints[0] = (0.5, 0.1, 0.1)   # nose, not confident

    vision.draw_coco_skeleton(frame, keypoints, min_score=0.5)

    assert frame[50, 50].any()        # shoulder-shoulder line drawn
    assert not frame[10, 50].any()    # nose not drawn
