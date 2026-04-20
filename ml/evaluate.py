#!/usr/bin/env python3
"""Evaluate smart parking models with task-specific metrics.

Stage 2 classification is the primary evaluation path and reports:
  top-1 accuracy, precision, recall, F1, confusion matrix, and per-class support.

Stage 1 detection support remains available for optional detector experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO

DEFAULT_STAGE1_DATA = "ml/stage1.yaml"
DEFAULT_STAGE2_DATA = "stage2_data"
DEFAULT_LOG_DIR = "logs"
CLASS_NAMES = ("free", "occupied")
WEATHER_NAMES = ("sunny", "cloudy", "rainy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate smart parking models. Stage 2 classification is the default path."
    )
    parser.add_argument("--weights", help="Path to model checkpoint (.pt or .onnx).")
    parser.add_argument("--data", default=None, help="Stage 1 YAML or Stage 2 dataset root.")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--stage1", action="store_true", help="Evaluate a Stage 1 detector.")
    parser.add_argument("--stage2", action="store_true", help="Evaluate a Stage 2 classifier.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="WEIGHTS",
        help="Compare multiple checkpoints on the selected Stage 2 split.",
    )
    parser.add_argument(
        "--cross-dataset",
        metavar="DATASET_DIR",
        help="Evaluate a Stage 2 classifier on another dataset root with free/occupied subdirs.",
    )
    parser.add_argument(
        "--per-weather",
        action="store_true",
        help="Evaluate Stage 2 weather subdirs if the dataset root exposes sunny/cloudy/rainy.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Classifier threshold sweep target used for deployed occupied/free decisions.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep Stage 2 classifier occupied threshold on the selected split.",
    )
    return parser.parse_args()


def append_csv(rows: list[dict], log_dir: Path, filename: str) -> None:
    if not rows:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    output = log_dir / filename
    write_header = not output.exists()
    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict], title: str) -> None:
    if not rows:
        return
    print(f"\n{title}")
    headers = list(rows[0].keys())
    widths = {header: max(len(header), max(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    template = "  ".join(f"{{:<{widths[header]}}}" for header in headers)
    print(template.format(*headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(template.format(*[str(row.get(header, "")) for header in headers]))


def stage2_root(data: str | None) -> Path:
    root = Path(data or DEFAULT_STAGE2_DATA)
    if not root.exists():
        raise SystemExit(f"Stage 2 dataset root not found: {root}")
    return root


def stage1_data(data: str | None) -> str:
    path = Path(data or DEFAULT_STAGE1_DATA)
    if not path.exists():
        raise SystemExit(f"Stage 1 data YAML not found: {path}")
    return str(path)


def classification_samples(dataset_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise SystemExit(f"Expected class directory missing: {class_dir}")
        for suffix in ("*.jpg", "*.jpeg", "*.png"):
            for path in sorted(class_dir.glob(suffix)):
                samples.append((path, class_name))
    if not samples:
        raise SystemExit(f"No classification samples found in {dataset_dir}")
    return samples


def occupied_probability(result) -> float:
    names = {int(k): v for k, v in result.names.items()}
    occupied_index = next((idx for idx, name in names.items() if name == "occupied"), None)
    if occupied_index is None:
        top1 = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        return top1_conf if names.get(top1) == "occupied" else 0.0
    return float(result.probs.data[occupied_index])


def classify_dataset(
    weights: str,
    dataset_dir: Path,
    *,
    device: str,
    imgsz: int,
    threshold: float,
) -> dict[str, object]:
    model = YOLO(weights)
    y_true: list[str] = []
    y_pred: list[str] = []

    for image_path, expected in classification_samples(dataset_dir):
        result = model(str(image_path), device=device, imgsz=imgsz, verbose=False)[0]
        occ_prob = occupied_probability(result)
        predicted = "occupied" if occ_prob >= threshold else "free"
        y_true.append(expected)
        y_pred.append(predicted)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES),
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))
    return {
        "top1_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision[1]), 4),
        "recall": round(float(recall[1]), 4),
        "f1": round(float(f1[1]), 4),
        "support": {cls: int(count) for cls, count in zip(CLASS_NAMES, support)},
        "confusion_matrix": matrix.tolist(),
        "sample_count": len(y_true),
    }


def evaluate_stage1(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for Stage 1 evaluation.")
    model = YOLO(args.weights)
    metrics = model.val(
        data=stage1_data(args.data),
        split=args.split,
        device=args.device,
        imgsz=max(args.imgsz, 640),
        verbose=False,
    ).results_dict
    row = {
        "model": Path(args.weights).parent.parent.name,
        "split": args.split,
        "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
    }
    print_rows([row], "Stage 1 Detection Evaluation")
    append_csv([row], Path(args.log_dir), "stage1_evaluation.csv")


def evaluate_stage2(weights: str, dataset_dir: Path, args: argparse.Namespace, label: str) -> dict[str, object]:
    metrics = classify_dataset(
        weights,
        dataset_dir,
        device=args.device,
        imgsz=args.imgsz,
        threshold=args.confidence_threshold,
    )
    return {
        "model": Path(weights).parent.parent.name if Path(weights).exists() else Path(weights).name,
        "dataset": label,
        "threshold": args.confidence_threshold,
        "top1_accuracy": metrics["top1_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "sample_count": metrics["sample_count"],
        "support_free": metrics["support"]["free"],
        "support_occupied": metrics["support"]["occupied"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }


def eval_split_dir(root: Path, split: str) -> Path:
    split_dir = root / split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found: {split_dir}")
    return split_dir


def evaluate_compare(args: argparse.Namespace) -> None:
    root = stage2_root(args.data)
    dataset_dir = eval_split_dir(root, args.split)
    rows = []
    for weights in args.compare:
        if not Path(weights).exists():
            print(f"Skipping missing checkpoint: {weights}")
            continue
        row = evaluate_stage2(weights, dataset_dir, args, f"{root.name}/{args.split}")
        row["size_mb"] = round(Path(weights).stat().st_size / 1_048_576, 2)
        rows.append(row)
    print_rows(rows, "Stage 2 Model Comparison")
    append_csv(rows, Path(args.log_dir), "stage2_model_comparison.csv")


def evaluate_cross_dataset(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for cross-dataset evaluation.")
    dataset_dir = Path(args.cross_dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Cross-dataset directory not found: {dataset_dir}")
    row = evaluate_stage2(args.weights, dataset_dir, args, dataset_dir.name)
    print_rows([row], "Stage 2 Cross-Dataset Evaluation")
    append_csv([row], Path(args.log_dir), "stage2_cross_dataset.csv")


def evaluate_per_weather(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for per-weather evaluation.")
    root = stage2_root(args.data)
    rows = []
    for weather in WEATHER_NAMES:
        weather_dir = root / weather
        if not weather_dir.exists():
            raise SystemExit(
                "Per-weather evaluation requested but weather labels are not exposed in "
                f"{root}. Expected <root>/<weather>/<class>/*.jpg for {', '.join(WEATHER_NAMES)}."
            )
        rows.append(evaluate_stage2(args.weights, weather_dir, args, weather))
    print_rows(rows, "Stage 2 Per-Weather Evaluation")
    append_csv(rows, Path(args.log_dir), "stage2_per_weather.csv")


def evaluate_threshold_sweep(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for threshold sweep.")
    root = stage2_root(args.data)
    dataset_dir = eval_split_dir(root, args.split)
    rows = []
    best_row = None
    for step in range(10, 95, 5):
        args.confidence_threshold = round(step / 100, 2)
        row = evaluate_stage2(args.weights, dataset_dir, args, f"{root.name}/{args.split}")
        rows.append(row)
        if best_row is None or row["f1"] > best_row["f1"]:
            best_row = dict(row)
    print_rows([best_row] if best_row else [], "Best Threshold Sweep Result")
    append_csv(rows, Path(args.log_dir), "stage2_threshold_sweep.csv")


def main() -> None:
    args = parse_args()
    if args.compare:
        evaluate_compare(args)
        return

    stage2_mode = not args.stage1
    if args.sweep:
        evaluate_threshold_sweep(args)
        return
    if args.cross_dataset:
        evaluate_cross_dataset(args)
        return
    if args.per_weather:
        evaluate_per_weather(args)
        return
    if not args.weights:
        raise SystemExit("--weights is required unless using --compare")
    if stage2_mode:
        root = stage2_root(args.data)
        row = evaluate_stage2(args.weights, eval_split_dir(root, args.split), args, f"{root.name}/{args.split}")
        print_rows([row], "Stage 2 Classification Evaluation")
        append_csv([row], Path(args.log_dir), "stage2_evaluation.csv")
        return
    evaluate_stage1(args)


if __name__ == "__main__":
    main()
