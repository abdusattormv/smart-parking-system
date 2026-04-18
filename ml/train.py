#!/usr/bin/env python3
"""Train YOLOv8-cls on clf-data parking spot images.

clf-data contains pre-cropped parking spot images (empty / not_empty),
so we use YOLOv8 image classification (-cls) rather than object detection.

Usage:
  python ml/train.py                             # YOLOv8n-cls, 50 epochs, mps
  python ml/train.py --variant s                 # YOLOv8s-cls
  python ml/train.py --variant m --device cpu    # on CPU / Colab
  python ml/train.py --variant n --epochs 10     # quick smoke test
  python ml/train.py --device cuda               # NVIDIA GPU
"""

import argparse
import json
import sys
import time
from pathlib import Path

import ultralytics
from ultralytics import YOLO

DEFAULT_DATA_YAML = "ml/data.yaml"
DEFAULT_EPOCHS    = 50
DEFAULT_IMGSZ     = 64      # clf-data crops are small; 64 px >> 15 px used by SVM
DEFAULT_BATCH     = 64      # classification is lighter than detection
DEFAULT_LR        = 0.001
DEFAULT_PATIENCE  = 10
PROJECT_DIR       = "runs/parking_clf"
MODEL_DIR         = "models"
MIN_ULTRALYTICS   = (8, 4, 38)


def _check_version() -> None:
    installed = tuple(int(x) for x in ultralytics.__version__.split(".")[:3])
    if installed < MIN_ULTRALYTICS:
        v = ".".join(str(x) for x in MIN_ULTRALYTICS)
        print(
            f"[warn] ultralytics {ultralytics.__version__} detected; "
            f"{v}+ recommended.\n"
            f"       pip install -U ultralytics",
            file=sys.stderr,
        )


def _train_with_fallback(model: YOLO, kwargs: dict, args: argparse.Namespace):
    """Run training; halve batch on MPS shape-mismatch error and retry once."""
    try:
        return model.train(**kwargs)
    except RuntimeError as exc:
        mps_crash = (
            args.device == "mps"
            and not getattr(args, "no_batch_fallback", False)
            and "shape mismatch" in str(exc)
        )
        if not mps_crash:
            raise
        new_batch = kwargs["batch"] // 2
        print(
            f"\n[MPS fallback] shape-mismatch → retrying with batch={new_batch}",
            file=sys.stderr,
        )
        kwargs = {**kwargs, "batch": new_batch, "exist_ok": True}
        return YOLO(model.ckpt_path).train(**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8-cls on clf-data (empty / not_empty)."
    )
    p.add_argument("--variant",  choices=["n", "s", "m"], default="n",
                   help="Model size: n (nano), s (small), m (medium).")
    p.add_argument("--data",     default=DEFAULT_DATA_YAML)
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",    type=int,   default=DEFAULT_IMGSZ,
                   help="Input image size. 64 works well for small spot crops.")
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR, dest="lr0")
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--device",   default="mps", help="cpu | cuda | mps | 0")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--no-batch-fallback", action="store_true")
    p.add_argument("--model-dir", default=MODEL_DIR,
                   help="Directory to save yolo_report.json.")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    data_yaml = Path(args.data)

    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}.\n"
            "Run: python ml/prepare_dataset.py"
        )

    _check_version()

    weights  = f"yolov8{args.variant}-cls.pt"
    run_name = f"yolov8{args.variant}_clf_parking"

    print(f"Model  : {weights}  (classification)")
    print(f"Data   : {data_yaml}")
    print(f"Device : {args.device}")
    print(f"Epochs : {args.epochs}  imgsz={args.imgsz}  batch={args.batch}\n")

    model = YOLO(weights)

    train_kwargs = dict(
        data      = str(data_yaml),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        optimizer = "AdamW",
        lr0       = args.lr0,
        patience  = args.patience,
        device    = args.device,
        project   = PROJECT_DIR,
        name      = run_name,
        resume    = args.resume,
        verbose   = True,
        deterministic = False,   # avoids MPS shape-mismatch in some builds
    )

    t0      = time.perf_counter()
    results = _train_with_fallback(model, train_kwargs, args)
    elapsed = time.perf_counter() - t0

    best_pt = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    print(f"\nTraining complete  ({elapsed:.0f}s)")
    print(f"Best checkpoint : {best_pt}")

    # Metrics — key names differ between ultralytics versions
    rd    = results.results_dict
    top1  = rd.get("metrics/accuracy_top1", rd.get("val/acc_top1"))
    top5  = rd.get("metrics/accuracy_top5", rd.get("val/acc_top5"))
    top1_str = f"{top1:.4f}" if isinstance(top1, float) else str(top1)
    top5_str = f"{top5:.4f}" if isinstance(top5, float) else str(top5)
    print(f"Top-1 accuracy  : {top1_str}")
    print(f"Top-5 accuracy  : {top5_str}")

    # Save JSON report (consumed by compare.py)
    mdl_dir = Path(args.model_dir)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model":         f"YOLOv8{args.variant}-cls",
        "weights":       weights,
        "epochs":        args.epochs,
        "imgsz":         args.imgsz,
        "batch":         args.batch,
        "train_time_s":  round(elapsed, 2),
        "best_ckpt":     str(best_pt),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
    }
    report_path = mdl_dir / "yolo_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()