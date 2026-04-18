#!/usr/bin/env python3
"""Train YOLOv8 variants on the PKLot dataset.

Usage:
  python ml/train.py                          # YOLOv8n, 50 epochs, mps
  python ml/train.py --variant s              # YOLOv8s
  python ml/train.py --variant m --device cpu # YOLOv8m on CPU (or Colab)
  python ml/train.py --variant n --epochs 10  # quick smoke test
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA_YAML = "ml/data.yaml"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_LR = 0.01
DEFAULT_PATIENCE = 10
PROJECT_DIR = "runs/parking"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on PKLot.")
    p.add_argument(
        "--variant",
        choices=["n", "s", "m"],
        default="n",
        help="YOLOv8 variant to train (n/s/m).",
    )
    p.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to data.yaml.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR, dest="lr0")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument(
        "--device",
        default="mps",
        help="Training device: mps, cpu, or cuda device index.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}.\n"
            "Run: python ml/prepare_dataset.py --pklot-dir datasets/pklot_raw"
        )

    weights = f"yolov8{args.variant}.pt"
    run_name = f"yolov8{args.variant}_pklot"
    print(f"Training {weights} for {args.epochs} epochs on {args.device}")

    model = YOLO(weights)
    results = model.train(
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
    )

    best_pt = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    print(f"\nTraining complete. Best checkpoint: {best_pt}")
    print(f"mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
