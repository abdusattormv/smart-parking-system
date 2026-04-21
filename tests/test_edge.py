import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.detect import (
    SmoothingBuffer,
    build_payload,
    classify_patch,
    crop_patch,
    resolve_camera_source,
    load_rois,
    normalize_rois,
    run_pipeline,
)


class FakeProbs:
    def __init__(self, top1: int, top1conf: float):
        self.top1 = top1
        self.top1conf = top1conf


class FakeResult:
    def __init__(self, label: str, confidence: float):
        self.names = {0: "free", 1: "occupied"}
        self.probs = FakeProbs(1 if label == "occupied" else 0, confidence)


class FakeYOLO:
    def __init__(self, labels):
        self.labels = iter(labels)

    def __call__(self, *_args, **_kwargs):
        label, confidence = next(self.labels)
        return [FakeResult(label, confidence)]


class FakeCapture:
    def __init__(self, opens: bool, reads: bool):
        self._opens = opens
        self._reads = reads
        self.released = False

    def isOpened(self):
        return self._opens

    def read(self):
        return self._reads, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_normalize_rois_accepts_valid_boxes():
    rois = normalize_rois({"spot_1": [0, 1, 10, 20]})
    assert rois == {"spot_1": (0, 1, 10, 20)}


def test_load_rois_reads_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("rois:\n  a: [1, 2, 3, 4]\n", encoding="utf-8")
    assert load_rois(config) == {"a": (1, 2, 3, 4)}


def test_resolve_camera_source_accepts_numeric_indexes():
    assert resolve_camera_source("0") == 0
    assert resolve_camera_source("12") == 12


def test_resolve_camera_source_rejects_invalid_tokens():
    with pytest.raises(ValueError, match="Invalid --camera value"):
        resolve_camera_source("front-door")


def test_resolve_camera_source_rejects_iphone_outside_macos(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Linux")
    with pytest.raises(ValueError, match="supported only on macOS"):
        resolve_camera_source("iphone")


def test_resolve_camera_source_requires_detected_iphone_camera(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"SPCameraDataType": [{"_name": "FaceTime HD Camera"}]}'),
    )
    with pytest.raises(ValueError, match="No iPhone camera detected"):
        resolve_camera_source("iphone")


def test_resolve_camera_source_uses_matching_iphone_camera(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPCameraDataType": [{"_name": "John’s iPhone Camera"}]}'
        ),
    )
    monkeypatch.setattr("edge.detect._macos_camera_backend", lambda: 1200)

    attempted = []

    def fake_video_capture(index, backend):
        attempted.append((index, backend))
        return FakeCapture(opens=index == 2, reads=index == 2)

    monkeypatch.setattr("edge.detect.cv2.VideoCapture", fake_video_capture)

    assert resolve_camera_source("iphone", probe_limit=4) == 2
    assert attempted == [(0, 1200), (1, 1200), (2, 1200)]


def test_resolve_camera_source_fails_when_no_camera_index_opens(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPCameraDataType": [{"_name": "Continuity Camera"}]}'
        ),
    )
    monkeypatch.setattr("edge.detect._macos_camera_backend", lambda: 1200)
    monkeypatch.setattr(
        "edge.detect.cv2.VideoCapture",
        lambda index, backend: FakeCapture(opens=False, reads=False),
    )

    with pytest.raises(RuntimeError, match="could not open any AVFoundation camera index"):
        resolve_camera_source("iphone", probe_limit=3)


def test_crop_patch_returns_none_for_invalid_box():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert crop_patch(frame, (10, 10, 5, 5)) is None


def test_classify_patch_thresholds_occupied_predictions():
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    model = FakeYOLO([("occupied", 0.49), ("occupied", 0.82)])

    first_status, first_conf = classify_patch(frame, (0, 0, 20, 20), model, "cpu", 0.5)
    second_status, second_conf = classify_patch(frame, (0, 0, 20, 20), model, "cpu", 0.5)

    assert (first_status, round(first_conf, 2)) == ("free", 0.49)
    assert (second_status, round(second_conf, 2)) == ("occupied", 0.82)


def test_classify_patch_invalid_crop_falls_back_to_free():
    frame = np.ones((40, 40, 3), dtype=np.uint8)
    status, confidence = classify_patch(
        frame,
        (100, 100, 120, 120),
        FakeYOLO([("occupied", 0.9)]),
        "cpu",
        0.5,
    )
    assert status == "free"
    assert confidence == 0.0


def test_smoothing_buffer_majority_vote():
    buffer = SmoothingBuffer(window=3)
    buffer.update({"spot_1": "occupied"})
    buffer.update({"spot_1": "free"})
    buffer.update({"spot_1": "occupied"})
    assert buffer.get_status()["spot_1"] == "occupied"


def test_smoothing_buffer_tie_resolves_to_free():
    buffer = SmoothingBuffer(window=4)
    for value in ("occupied", "free", "occupied", "free"):
        buffer.update({"spot_1": value})
    assert buffer.get_status()["spot_1"] == "free"


def test_build_payload_uses_v3_schema():
    payload = build_payload({"spot_1": "free"}, {"spot_1": 0.8123})
    assert payload["spots"] == {"spot_1": "free"}
    assert payload["confidence"] == {"spot_1": 0.812}
    assert payload["timestamp"].endswith("Z")


def test_run_pipeline_uses_shared_postprocess_path():
    frame = np.ones((60, 60, 3), dtype=np.uint8)
    args = type(
        "Args",
        (),
        {"device": "cpu", "stage1_detector": False, "stage2_threshold": 0.5},
    )()

    payload, spot_boxes = run_pipeline(
        frame=frame,
        fixed_rois={"spot_1": (0, 0, 20, 20), "spot_2": (20, 0, 40, 20)},
        stage1_model=None,
        stage2_model=FakeYOLO([("occupied", 0.9), ("free", 0.7)]),
        smooth_buf=SmoothingBuffer(window=1),
        args=args,
    )

    assert spot_boxes["spot_1"] == (0, 0, 20, 20)
    assert payload["spots"] == {"spot_1": "occupied", "spot_2": "free"}
    assert payload["confidence"] == {"spot_1": 0.9, "spot_2": 0.7}
