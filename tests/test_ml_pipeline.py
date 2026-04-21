import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import prepare_dataset, train


def make_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(color, color, color)).save(path)


def make_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def test_prepare_single_model_detection_preserves_two_classes(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "sample.jpg", 20)
    make_label(
        root / "train" / "labels" / "sample.txt",
        [
            "0 0.5 0.5 0.4 0.4",
            "1 0.2 0.2 0.1 0.1",
        ],
    )
    make_image(root / "valid" / "images" / "sample.jpg", 20)
    make_label(root / "valid" / "labels" / "sample.txt", ["1 0.5 0.5 0.4 0.4"])
    make_image(root / "test" / "images" / "sample.jpg", 20)
    make_label(root / "test" / "labels" / "sample.txt", ["0 0.5 0.5 0.4 0.4"])

    out_dir = tmp_path / "single_model_data"
    yaml_path = tmp_path / "single_model.yaml"
    prepare_dataset.prepare_single_model_detection(root, out_dir, yaml_path)

    assert (out_dir / "train" / "images" / "sample.jpg").exists()
    labels = (out_dir / "train" / "labels" / "sample.txt").read_text(encoding="utf-8").splitlines()
    assert labels == ["0 0.5 0.5 0.4 0.4", "1 0.2 0.2 0.1 0.1"]
    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "names:" in yaml_text
    assert "- free" in yaml_text
    assert "- occupied" in yaml_text
    report = json.loads((out_dir / prepare_dataset.DETECTION_REPORT).read_text(encoding="utf-8"))
    assert report["track"] == "single_model"
    assert report["splits"]["train"]["empty_label_frames_excluded"] == 0


def test_prepare_stage1_excludes_empty_label_frames(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "kept.jpg", 20)
    make_label(root / "train" / "labels" / "kept.txt", ["1 0.5 0.5 0.4 0.4"])
    make_image(root / "train" / "images" / "empty.jpg", 20)
    make_label(root / "train" / "labels" / "empty.txt", [])
    make_image(root / "valid" / "images" / "sample.jpg", 20)
    make_label(root / "valid" / "labels" / "sample.txt", ["0 0.5 0.5 0.4 0.4"])
    make_image(root / "test" / "images" / "sample.jpg", 20)
    make_label(root / "test" / "labels" / "sample.txt", ["0 0.5 0.5 0.4 0.4"])

    out_dir = tmp_path / "stage1_data"
    yaml_path = tmp_path / "stage1.yaml"
    prepare_dataset.prepare_stage1(root, out_dir, yaml_path)

    assert (out_dir / "train" / "images" / "kept.jpg").exists()
    assert not (out_dir / "train" / "images" / "empty.jpg").exists()
    report = json.loads((out_dir / prepare_dataset.DETECTION_REPORT).read_text(encoding="utf-8"))
    assert report["track"] == "stage1"
    assert report["splits"]["train"]["empty_label_frames_excluded"] == 1


def test_train_requires_explicit_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    with pytest.raises(SystemExit) as exc:
        train.parse_args()
    assert exc.value.code == 2


def test_train_stage2_mode_resolution(monkeypatch, tmp_path):
    data_dir = tmp_path / "stage2_data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "STAGE2_DATA_DIR", str(data_dir))
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage2"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["task"] == "classify"
    assert defaults["track"] == "stage2"
    assert defaults["data_path"] == str(data_dir)


def test_train_single_model_mode_resolution(monkeypatch, tmp_path):
    yaml_path = tmp_path / "single_model.yaml"
    yaml_path.write_text("path: /tmp\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: [free, occupied]\n", encoding="utf-8")
    monkeypatch.setattr(train, "SINGLE_MODEL_YAML", str(yaml_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--single-model"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["task"] == "detect"
    assert defaults["track"] == "single_model"
    assert defaults["project_dir"] == train.SINGLE_MODEL_PROJECT


def test_train_stage2_accuracy_defaults(monkeypatch, tmp_path):
    data_dir = tmp_path / "stage2_data"
    data_dir.mkdir()
    monkeypatch.setattr(train, "STAGE2_DATA_DIR", str(data_dir))
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage2"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["lr0"] == train.STAGE2_LR
    assert defaults["patience"] == train.STAGE2_PATIENCE
    assert defaults["dropout"] == 0.1
    assert defaults["cos_lr"] is True


def test_existing_checkpoint_prefers_best_then_last(tmp_path):
    best_ckpt, last_ckpt = train._checkpoint_paths(str(tmp_path / "runs"), "exp")
    last_ckpt.parent.mkdir(parents=True, exist_ok=True)

    assert train._existing_checkpoint(best_ckpt, last_ckpt) is None

    last_ckpt.write_bytes(b"last")
    assert train._existing_checkpoint(best_ckpt, last_ckpt) == last_ckpt

    best_ckpt.write_bytes(b"best")
    assert train._existing_checkpoint(best_ckpt, last_ckpt) == best_ckpt


def test_nan_recovery_patch_raises_clear_error_when_last_missing(tmp_path):
    train._patch_ultralytics_trainer_for_nan_checkpoints()

    fake_trainer = SimpleNamespace(
        loss=torch.tensor(float("nan")),
        fitness=float("nan"),
        best_fitness=0.0,
        start_epoch=0,
        last=tmp_path / "weights" / "last.pt",
        nan_recovery_attempts=0,
    )

    with pytest.raises(RuntimeError) as exc:
        train.BaseTrainer._handle_nan_recovery(fake_trainer, epoch=0)

    message = str(exc.value)
    assert "before a recoverable checkpoint was written" in message
    assert str(fake_trainer.last) in message
