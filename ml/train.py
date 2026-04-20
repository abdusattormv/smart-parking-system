#!/usr/bin/env python3
"""Train YOLOv8 models for the smart parking v3 pipeline.

Primary path:
  Stage 2 patch classification with YOLOv8*-cls.

Secondary path:
  Stage 1 spot detection for optional ROI discovery experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ultralytics
from ultralytics import YOLO

STAGE1_YAML = "ml/stage1.yaml"
STAGE2_DATA_DIR = "stage2_data"

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 60
STAGE1_IMGSZ = 640
STAGE2_IMGSZ = 64
DEFAULT_BATCH = 16
STAGE2_BATCH = 64
DEFAULT_LR = 0.003
DEFAULT_PATIENCE = 15
STAGE1_PROJECT = "runs/stage1_det"
STAGE2_PROJECT = "runs/stage2_cls"
MODEL_DIR = "models"
MIN_ULTRALYTICS = (8, 4, 38)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 classification by default, or Stage 1 detection when requested."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage1", action="store_true", help="Train Stage 1 detector.")
    group.add_argument("--stage2", action="store_true", help="Train Stage 2 classifier.")
    parser.add_argument("--variant", choices=["n", "s", "m"], default="n")
    parser.add_argument("--data", default=None, help="Override data path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, dest="lr0")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-batch-fallback", action="store_true")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--erasing", type=float, default=0.2)
    parser.add_argument("--mixup", type=float, default=0.0)
    return parser.parse_args()


def _check_version() -> None:
    version = tuple(int(part) for part in ultralytics.__version__.split(".")[:3])
    if version < MIN_ULTRALYTICS:
        required = ".".join(str(part) for part in MIN_ULTRALYTICS)
        print(
            f"[warn] ultralytics {ultralytics.__version__} detected; {required}+ recommended.",
            file=sys.stderr,
        )


def _train_with_fallback(model: YOLO, kwargs: dict, args: argparse.Namespace):
    try:
        return model.train(**kwargs)
    except RuntimeError as exc:
        if args.device != "mps" or args.no_batch_fallback or "shape mismatch" not in str(exc):
            raise
        new_batch = max(1, kwargs["batch"] // 2)
        print(f"[MPS fallback] retrying with batch={new_batch}", file=sys.stderr)
        retry_kwargs = {**kwargs, "batch": new_batch, "exist_ok": True}
        return YOLO(model.ckpt_path).train(**retry_kwargs)


def resolve_stage1_data(data: str | None) -> str:
    path = Path(data or STAGE1_YAML)
    if not path.exists():
        raise SystemExit(
            f"Stage 1 YAML not found: {path}\n"
            "Run: python ml/prepare_dataset.py --stage1 --pklot-dir <roboflow-export>"
        )
    return str(path)


def resolve_stage2_data(data: str | None) -> str:
    path = Path(data or STAGE2_DATA_DIR)
    if not path.exists():
        raise SystemExit(
            f"Stage 2 data not found: {path}\n"
            "Run: python ml/prepare_dataset.py --stage2 --pklot-dir <roboflow-export> "
            "[--cnrpark-dir <cnrpatches>]"
        )
    return str(path)


def task_defaults(args: argparse.Namespace) -> dict[str, object]:
    stage2 = not args.stage1
    if stage2:
        return {
            "task": "classify",
            "data_path": resolve_stage2_data(args.data),
            "weights": f"yolov8{args.variant}-cls.pt",
            "epochs": args.epochs or STAGE2_EPOCHS,
            "imgsz": args.imgsz or STAGE2_IMGSZ,
            "batch": args.batch or STAGE2_BATCH,
            "project_dir": STAGE2_PROJECT,
            "run_name": f"yolov8{args.variant}_stage2",
            "report_name": f"stage2_{args.variant}_report.json",
        }
    return {
        "task": "detect",
        "data_path": resolve_stage1_data(args.data),
        "weights": f"yolov8{args.variant}.pt",
        "epochs": args.epochs or STAGE1_EPOCHS,
        "imgsz": args.imgsz or STAGE1_IMGSZ,
        "batch": args.batch or DEFAULT_BATCH,
        "project_dir": STAGE1_PROJECT,
        "run_name": f"yolov8{args.variant}_stage1",
        "report_name": f"stage1_{args.variant}_report.json",
    }


def extract_metrics(task: str, results) -> dict[str, object]:
    metrics = results.results_dict
    if task == "classify":
        top1 = metrics.get("metrics/accuracy_top1", metrics.get("val/acc_top1"))
        top5 = metrics.get("metrics/accuracy_top5", metrics.get("val/acc_top5"))
        return {"top1_accuracy": top1, "top5_accuracy": top5}
    return {
        "mAP50": metrics.get("metrics/mAP50(B)", metrics.get("val/map50")),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", metrics.get("val/map")),
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
    }


def main() -> None:
    args = parse_args()
    _check_version()
    defaults = task_defaults(args)

    print(f"Task   : {defaults['task']}")
    print(f"Model  : {defaults['weights']}")
    print(f"Data   : {defaults['data_path']}")
    print(f"Device : {args.device}")
    print(
        f"Epochs : {defaults['epochs']}  imgsz={defaults['imgsz']}  "
        f"batch={defaults['batch']}"
    )

    model = YOLO(defaults["weights"])
    train_kwargs = {
        "data": defaults["data_path"],
        "epochs": defaults["epochs"],
        "imgsz": defaults["imgsz"],
        "batch": defaults["batch"],
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "patience": args.patience,
        "device": args.device,
        "project": defaults["project_dir"],
        "name": defaults["run_name"],
        "resume": args.resume,
        "verbose": True,
        "deterministic": False,
        "degrees": args.degrees,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "scale": args.scale,
        "erasing": args.erasing,
        "mixup": args.mixup,
    }

    started_at = time.perf_counter()
    results = _train_with_fallback(model, train_kwargs, args)
    elapsed = time.perf_counter() - started_at

    best_ckpt = Path(defaults["project_dir"]) / defaults["run_name"] / "weights" / "best.pt"
    metric_report = extract_metrics(defaults["task"], results)
    report = {
        "task": defaults["task"],
        "variant": args.variant,
        "model": defaults["weights"],
        "data": defaults["data_path"],
        "epochs": defaults["epochs"],
        "imgsz": defaults["imgsz"],
        "batch": defaults["batch"],
        "lr0": args.lr0,
        "patience": args.patience,
        "augmentation": {
            "degrees": args.degrees,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
            "scale": args.scale,
            "erasing": args.erasing,
            "mixup": args.mixup,
        },
        "train_time_s": round(elapsed, 2),
        "best_ckpt": str(best_ckpt),
        **metric_report,
    }

    print(f"\nTraining complete ({elapsed:.0f}s)")
    print(f"Best checkpoint : {best_ckpt}")
    if defaults["task"] == "classify":
        print(f"Top-1 accuracy : {metric_report.get('top1_accuracy')}")
        print(f"Top-5 accuracy : {metric_report.get('top5_accuracy')}")
    else:
        print(f"mAP@50         : {metric_report.get('mAP50')}")
        print(f"mAP@50-95      : {metric_report.get('mAP50_95')}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / str(defaults["report_name"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved   : {report_path}")
    if defaults["task"] == "classify":
        print("Comparison set : run variants n, s, m and compare Top-1 accuracy, size, and latency.")


if __name__ == "__main__":
    main()
