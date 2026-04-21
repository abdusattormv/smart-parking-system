import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge import benchmark


def test_clip_roi_returns_original_frame_when_absent():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    assert benchmark.clip_roi(frame, None).shape == frame.shape


def test_clip_roi_clips_out_of_bounds_coordinates():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    cropped = benchmark.clip_roi(frame, (-5, -1, 50, 8))
    assert cropped.shape == (8, 20, 3)


def test_prepare_runtime_frame_resizes_classifier_input():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    prepared = benchmark.prepare_runtime_frame(
        frame,
        task="classify",
        imgsz=64,
        roi=(10, 20, 50, 80),
    )
    assert prepared.shape == (64, 64, 3)


def test_prepare_runtime_frame_keeps_detector_frame_shape():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    prepared = benchmark.prepare_runtime_frame(
        frame,
        task="detect",
        imgsz=768,
        roi=(10, 20, 50, 80),
    )
    assert prepared.shape == (60, 40, 3)
