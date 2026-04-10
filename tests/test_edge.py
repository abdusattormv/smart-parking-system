import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.detect import DEFAULT_OVERLAP_THRESHOLD, SmoothingBuffer, spot_occupied


# ---------------------------------------------------------------------------
# spot_occupied
# ---------------------------------------------------------------------------

def test_spot_occupied_no_boxes():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert spot_occupied([], roi) is False


def test_spot_occupied_overlap():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # Box fully inside ROI: intersection 80×80 = 6400, roi_area 10000 → ratio 0.64 > 0.2
    boxes = [[10.0, 10.0, 90.0, 90.0]]
    assert spot_occupied(boxes, roi, threshold=DEFAULT_OVERLAP_THRESHOLD) is True


def test_spot_occupied_no_overlap():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # Box completely outside
    boxes = [[200.0, 200.0, 300.0, 300.0]]
    assert spot_occupied(boxes, roi) is False


def test_spot_occupied_partial_overlap_below_threshold():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # Box overlaps only 5×100 = 500 / 10000 = 5% < 20% threshold
    boxes = [[-95.0, 0.0, 5.0, 100.0]]
    assert spot_occupied(boxes, roi, threshold=0.2) is False


def test_spot_occupied_partial_overlap_above_threshold():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # Box overlaps 50×100 = 5000 / 10000 = 50% > 20% threshold
    boxes = [[-50.0, 0.0, 50.0, 100.0]]
    assert spot_occupied(boxes, roi, threshold=0.2) is True


def test_spot_occupied_zero_threshold():
    """threshold=0.0 accepts any overlap (matches PRD's simple existence check)."""
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    boxes = [[-95.0, 0.0, 5.0, 100.0]]  # only 5% overlap
    assert spot_occupied(boxes, roi, threshold=0.0) is True


def test_spot_occupied_multiple_boxes_one_qualifies():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    boxes = [
        [200.0, 200.0, 300.0, 300.0],  # outside
        [10.0, 10.0, 90.0, 90.0],      # inside (64% overlap)
    ]
    assert spot_occupied(boxes, roi) is True


# ---------------------------------------------------------------------------
# SmoothingBuffer
# ---------------------------------------------------------------------------

def test_smoothing_majority_occupied():
    buf = SmoothingBuffer(["A"], window=5)
    for v in [True, True, True, False, False]:  # 3 vs 2 → occupied
        buf.update({"A": v})
    assert buf.get_status()["A"] == "occupied"


def test_smoothing_all_free():
    buf = SmoothingBuffer(["A"], window=3)
    for v in [False, False, False]:
        buf.update({"A": v})
    assert buf.get_status()["A"] == "free"


def test_smoothing_single_true_frame():
    # 1 reading in a window-5 buffer: sum=1 > len=1/2=0.5 → occupied
    buf = SmoothingBuffer(["A"], window=5)
    buf.update({"A": True})
    assert buf.get_status()["A"] == "occupied"


def test_smoothing_single_false_frame():
    buf = SmoothingBuffer(["A"], window=5)
    buf.update({"A": False})
    assert buf.get_status()["A"] == "free"


def test_smoothing_empty_buffer_defaults_free():
    buf = SmoothingBuffer(["A"], window=5)
    assert buf.get_status()["A"] == "free"


def test_smoothing_tie_resolves_to_free():
    # 50/50 tie: sum=2, len=4, 2 > 2.0 is False → "free"
    buf = SmoothingBuffer(["A"], window=4)
    for v in [True, True, False, False]:
        buf.update({"A": v})
    assert buf.get_status()["A"] == "free"


def test_smoothing_multiple_spots():
    buf = SmoothingBuffer(["s1", "s2"], window=3)
    for _ in range(3):
        buf.update({"s1": True, "s2": False})
    status = buf.get_status()
    assert status["s1"] == "occupied"
    assert status["s2"] == "free"


def test_smoothing_reset_clears_history():
    buf = SmoothingBuffer(["A"], window=3)
    for _ in range(3):
        buf.update({"A": True})
    buf.reset()
    assert buf.get_status()["A"] == "free"


def test_smoothing_unknown_spot_ignored():
    """update() with an unknown spot key should not raise."""
    buf = SmoothingBuffer(["A"], window=3)
    buf.update({"B": True})  # "B" not in buffer — silently ignored
    assert buf.get_status()["A"] == "free"
