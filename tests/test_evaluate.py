import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import evaluate


class FakeProbs:
    def __init__(self, occupied_prob: float, top1: int, top1conf: float):
        self.data = [1.0 - occupied_prob, occupied_prob]
        self.top1 = top1
        self.top1conf = top1conf


class FakeResult:
    def __init__(self, occupied_prob: float):
        self.names = {0: "free", 1: "occupied"}
        top1 = 1 if occupied_prob >= 0.5 else 0
        top1conf = occupied_prob if top1 == 1 else 1.0 - occupied_prob
        self.probs = FakeProbs(occupied_prob, top1, top1conf)


class FakeYOLO:
    probs_by_name = {"free_ok.jpg": 0.2, "occupied_ok.jpg": 0.8}

    def __init__(self, _weights: str):
        pass

    def __call__(self, image_path: str, **_kwargs):
        return [FakeResult(self.probs_by_name[Path(image_path).name])]


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(path)


def test_classify_dataset_reports_classification_metrics(tmp_path, monkeypatch):
    make_image(tmp_path / "val" / "free" / "free_ok.jpg")
    make_image(tmp_path / "val" / "occupied" / "occupied_ok.jpg")
    monkeypatch.setattr(evaluate, "YOLO", FakeYOLO)

    metrics = evaluate.classify_dataset(
        "fake.pt",
        tmp_path / "val",
        device="cpu",
        imgsz=64,
        threshold=0.5,
    )

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
