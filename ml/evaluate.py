#!/usr/bin/env python3
"""Evaluate a trained YOLOv8 model on the PKLot test set.

Modes:
  --full          Full test-set evaluation (mAP, precision, recall, F1)
  --per-weather   Separate evaluation for sunny / cloudy / rainy splits
  --sweep         Confidence and IoU NMS threshold sweep (finds best F1)
  --compare       Evaluate multiple model checkpoints and print comparison table

Output: logs/evaluation_results.csv

Usage:
  python ml/evaluate.py --weights artifacts/models/best.pt --full
  python ml/evaluate.py --weights artifacts/models/best.pt --per-weather
  python ml/evaluate.py --weights artifacts/models/best.pt --sweep
  python ml/evaluate.py --compare \
      runs/parking/yolov8n_pklot/weights/best.pt \
      runs/parking/yolov8s_pklot/weights/best.pt \
      runs/parking/yolov8m_pklot/weights/best.pt
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA_YAML = "ml/data.yaml"
DEFAULT_LOG_DIR = "logs"
DEFAULT_IMGSZ = 640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 model on PKLot.")
    p.add_argument("--weights", help="Path to model checkpoint (.pt or .onnx).")
    p.add_argument("--data", default=DEFAULT_DATA_YAML)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--device", default="mps")
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)

    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="Full test-set evaluation.")
    modes.add_argument("--per-weather", action="store_true", help="Per-weather evaluation.")
    modes.add_argument("--sweep", action="store_true", help="Threshold sweep.")
    modes.add_argument(
        "--compare",
        nargs="+",
        metavar="WEIGHTS",
        help="Compare multiple checkpoints.",
    )
    return p.parse_args()


def run_val(model: YOLO, data: str, split: str, conf: float, iou: float, device: str, imgsz: int) -> dict:
    """Run model.val() and return a metrics dict."""
    metrics = model.val(
        data=data,
        split=split,
        conf=conf,
        iou=iou,
        device=device,
        imgsz=imgsz,
        verbose=False,
    )
    rd = metrics.results_dict
    return {
        "mAP50": round(rd.get("metrics/mAP50(B)", 0.0), 4),
        "mAP50_95": round(rd.get("metrics/mAP50-95(B)", 0.0), 4),
        "precision": round(rd.get("metrics/precision(B)", 0.0), 4),
        "recall": round(rd.get("metrics/recall(B)", 0.0), 4),
        "f1": round(
            2
            * rd.get("metrics/precision(B)", 0.0)
            * rd.get("metrics/recall(B)", 0.0)
            / max(rd.get("metrics/precision(B)", 0.0) + rd.get("metrics/recall(B)", 0.0), 1e-9),
            4,
        ),
    }


def print_table(rows: list[dict], title: str = "") -> None:
    if title:
        print(f"\n{'='*60}")
        print(title)
        print("=" * 60)
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    fmt = "  ".join(f"{{:<{widths[h]}}}" for h in headers)
    print(fmt.format(*headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(fmt.format(*[str(row.get(h, "")) for h in headers]))


def append_csv(rows: list[dict], log_dir: Path, filename: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / filename
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults appended to {out}")


def eval_full(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    print(f"Evaluating {args.weights} on full test set ...")
    m = run_val(model, args.data, "test", conf=0.25, iou=0.45, device=args.device, imgsz=args.imgsz)
    model_name = Path(args.weights).parent.parent.name
    row = {"model": model_name, "split": "test", "conf": 0.25, "iou": 0.45, **m}
    print_table([row], "Full Test Set Results")

    # Save confusion matrix and PR curve plots (Ultralytics outputs to runs/)
    log_dir = Path(args.log_dir)
    append_csv([row], log_dir, "evaluation_results.csv")


def eval_per_weather(args: argparse.Namespace) -> None:
    data_yaml = Path(args.data)
    import yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg["path"])

    model = YOLO(args.weights)
    model_name = Path(args.weights).parent.parent.name
    rows = []

    for weather in ["sunny", "cloudy", "rainy"]:
        split_dir = dataset_root / f"test_{weather}"
        if not split_dir.exists():
            print(f"  Skipping {weather} (no split at {split_dir})")
            continue

        # Build a temporary single-split data yaml pointing to this split
        import tempfile
        tmp_data = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": f"test_{weather}/images",
            "nc": data_cfg["nc"],
            "names": data_cfg["names"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            import yaml as _yaml
            _yaml.dump(tmp_data, tf)
            tmp_path = tf.name

        print(f"  Evaluating {weather} split ...")
        m = run_val(model, tmp_path, "test", conf=0.25, iou=0.45, device=args.device, imgsz=args.imgsz)
        rows.append({"model": model_name, "weather": weather, **m})
        Path(tmp_path).unlink(missing_ok=True)

    print_table(rows, "Per-Weather Evaluation")
    append_csv(rows, Path(args.log_dir), "evaluation_results.csv")


def eval_sweep(args: argparse.Namespace) -> None:
    import numpy as np

    model = YOLO(args.weights)
    print("Sweeping confidence and IoU thresholds ...")
    rows = []
    best_f1 = -1.0
    best_row = {}

    for conf in [round(x, 2) for x in list(range(10, 55, 5))]:
        conf /= 100
        for iou in [round(x, 2) for x in list(range(30, 75, 5))]:
            iou /= 100
            m = run_val(model, args.data, "val", conf=conf, iou=iou, device=args.device, imgsz=args.imgsz)
            row = {"conf": conf, "iou": iou, **m}
            rows.append(row)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_row = row

    print_table([best_row], f"Best operating point (F1={best_f1:.4f})")
    append_csv(rows, Path(args.log_dir), "threshold_sweep.csv")
    print(f"\nRecommended: --confidence-threshold {best_row['conf']} --iou-threshold {best_row['iou']}")


def eval_compare(args: argparse.Namespace) -> None:
    rows = []
    for weights_path in args.compare:
        p = Path(weights_path)
        if not p.exists():
            print(f"  Skipping {weights_path} (not found)")
            continue
        model = YOLO(str(p))
        model_name = p.parent.parent.name
        size_mb = round(p.stat().st_size / 1_048_576, 1)
        print(f"  Evaluating {model_name} ...")
        m = run_val(model, args.data, "test", conf=0.25, iou=0.45, device=args.device, imgsz=args.imgsz)
        rows.append({"model": model_name, "size_mb": size_mb, **m})

    print_table(rows, "Model Comparison")
    append_csv(rows, Path(args.log_dir), "model_comparison.csv")


def main() -> None:
    args = parse_args()

    if args.compare:
        eval_compare(args)
        return

    if not args.weights:
        raise SystemExit("--weights is required unless using --compare")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found: {weights}")

    if args.per_weather:
        eval_per_weather(args)
    elif args.sweep:
        eval_sweep(args)
    else:
        eval_full(args)


if __name__ == "__main__":
    main()
