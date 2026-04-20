import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import prepare_dataset, train


def make_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(color, color, color)).save(path)


def test_stratified_split_creates_all_splits():
    class_images = {
        "free": [Path(f"free_{i}.jpg") for i in range(10)],
        "occupied": [Path(f"occ_{i}.jpg") for i in range(10)],
    }
    splits = prepare_dataset.stratified_split(class_images, val_ratio=0.2, test_ratio=0.2, seed=7)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(splits[split]["free"] for split in splits)
    assert all(splits[split]["occupied"] for split in splits)


def test_copy_images_handles_filename_collisions(tmp_path):
    src_a = tmp_path / "src_a" / "free" / "image.jpg"
    src_b = tmp_path / "src_b" / "free" / "image.jpg"
    make_image(src_a, 10)
    make_image(src_b, 20)

    collisions = prepare_dataset.copy_images(
        {"train": {"free": [src_a, src_b]}},
        tmp_path / "out",
    )

    written = list((tmp_path / "out" / "train" / "free").glob("*.jpg"))
    assert len(written) == 2
    assert collisions["train/free"] == 1


def test_sanity_check_writes_report(tmp_path):
    src = tmp_path / "patches" / "free" / "a.jpg"
    make_image(src, 10)
    src2 = tmp_path / "patches" / "occupied" / "b.jpg"
    make_image(src2, 20)

    prepare_dataset.sanity_check_stage2(
        {
            "train": {"free": [src], "occupied": [src2]},
            "val": {"free": [src], "occupied": [src2]},
            "test": {"free": [src], "occupied": [src2]},
        },
        all_images={"free": [src], "occupied": [src2]},
        collisions={},
        report_path=tmp_path / "stage2_data" / "dataset_report.json",
    )

    report = json.loads((tmp_path / "stage2_data" / "dataset_report.json").read_text())
    assert report["class_counts"] == {"free": 1, "occupied": 1}
    assert report["dimension_summary"]["free"]["sampled"] == 1


def test_weather_split_paths_requires_expected_layout(tmp_path):
    try:
        prepare_dataset.weather_split_paths(tmp_path)
    except SystemExit as exc:
        assert "Expected weather layout" in str(exc)
    else:
        raise AssertionError("expected weather_split_paths to fail")


def test_train_stage2_is_default(monkeypatch, tmp_path):
    data_dir = tmp_path / "stage2_data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "STAGE2_DATA_DIR", str(data_dir))
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["task"] == "classify"
    assert defaults["data_path"] == str(data_dir)
