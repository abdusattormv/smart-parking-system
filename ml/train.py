#!/usr/bin/env python3
"""Train YOLOv8 variants on the PKLot dataset.

Usage:
  python ml/train.py                          # YOLOv8n, 50 epochs, mps
  python ml/train.py --variant s              # YOLOv8s
  python ml/train.py --variant m --device cpu # YOLOv8m on CPU (or Colab)
  python ml/train.py --variant n --epochs 10  # quick smoke test
"""

import argparse
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

DEFAULT_DATA_YAML = "ml/data.yaml"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_LR = 0.01
DEFAULT_PATIENCE = 10
PROJECT_DIR = "runs/parking"
MIN_ULTRALYTICS = (8, 4, 38)  # first version with MPS assigner fix


def _check_ultralytics_version() -> None:
    installed = tuple(int(x) for x in ultralytics.__version__.split(".")[:3])
    if installed < MIN_ULTRALYTICS:
        v = ".".join(str(x) for x in MIN_ULTRALYTICS)
        print(
            f"[warn] ultralytics {ultralytics.__version__} detected. "
            f"MPS shape-mismatch bug is fixed in {v}+.\n"
            f"       Run: pip install -U ultralytics",
            file=sys.stderr,
        )


def _train_with_fallback(model, kwargs: dict, args):
    """Run training; halve batch on MPS RuntimeError and retry once."""
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
            f"\n[MPS fallback] shape-mismatch caught — retrying with batch={new_batch}",
            file=sys.stderr,
        )
        kwargs = {**kwargs, "batch": new_batch, "exist_ok": True}
        return YOLO(model.ckpt_path).train(**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on PKLot.")
    p.add_argument("--variant", choices=["n", "s", "m"], default="n")
    p.add_argument("--data", default=DEFAULT_DATA_YAML)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR, dest="lr0")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--device", default="mps")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-batch-fallback", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}.\n"
            "Run: python ml/prepare_dataset.py --pklot-dir datasets/pklot_raw"
        )

    _check_ultralytics_version()

    weights = f"yolov8{args.variant}.pt"
    run_name = f"yolov8{args.variant}_pklot"
    print(f"Training {weights} for {args.epochs} epochs on {args.device}")

    model = YOLO(weights)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        project=PROJECT_DIR,
        name=run_name,
        resume=args.resume,
        verbose=True,
        deterministic=False,  # avoids MPS shape-mismatch in task-aligned assigner
    )
    results = _train_with_fallback(model, train_kwargs, args)

    best_pt = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    print(f"\nTraining complete. Best checkpoint: {best_pt}")
    print(f"mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()