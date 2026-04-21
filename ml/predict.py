#!/usr/bin/env python3
"""Run one-off prediction for smart parking models.

Primary path:
  Stage 2 patch classification with YOLOv8*-cls.

Optional paths:
  Stage 1 spot detection or the single-model occupancy detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

DEFAULT_STAGE2_IMGSZ = 64
DEFAULT_DETECT_IMGSZ = 640
DEFAULT_CONF = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with Stage 2 classification by default, or Stage 1 / single-model detection."
    )
    parser.add_argument("--weights", required=True, help="Path to model checkpoint (.pt or .onnx).")
    parser.add_argument("--source", required=True, help="Image path, directory, or video/camera source accepted by Ultralytics.")
    parser.add_argument("--device", default="mps", help="Inference device, e.g. mps or cpu.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size override.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Detection confidence threshold.")
    parser.add_argument("--save", action="store_true", help="Save Ultralytics prediction artifacts.")

    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 detection prediction.")
    parser.add_argument("--stage2", action="store_true", help="Run Stage 2 classification prediction.")
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Run the ML-only single-model occupancy detector prediction.",
    )
    return parser.parse_args()


def prediction_mode(args: argparse.Namespace) -> str:
    modes = [args.stage1, args.stage2, args.single_model]
    if sum(bool(value) for value in modes) > 1:
        raise SystemExit("Choose only one of --stage1, --stage2, or --single-model.")
    if args.stage1:
        return "stage1"
    if args.single_model:
        return "single_model"
    return "stage2"


def default_imgsz(mode: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    return DEFAULT_STAGE2_IMGSZ if mode == "stage2" else DEFAULT_DETECT_IMGSZ


def classification_output(result: Any, *, source: str, weights: str) -> dict[str, object]:
    top1 = int(result.probs.top1)
    label = str(result.names[top1])
    confidence = float(result.probs.top1conf)
    return {
        "task": "classify",
        "source": source,
        "weights": weights,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            str(name): round(float(result.probs.data[int(index)]), 4)
            for index, name in sorted(result.names.items(), key=lambda item: int(item[0]))
        },
    }


def detection_output(result: Any, *, source: str, weights: str, mode: str) -> dict[str, object]:
    boxes = []
    names = {int(index): str(name) for index, name in result.names.items()}
    xyxy = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [round(float(value), 2) for value in box]
        class_id = int(cls_id)
        boxes.append(
            {
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
                "confidence": round(float(conf), 4),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return {
        "task": "detect",
        "track": mode,
        "source": source,
        "weights": weights,
        "count": len(boxes),
        "boxes": boxes,
    }


def main() -> None:
    args = parse_args()
    mode = prediction_mode(args)
    imgsz = default_imgsz(mode, args.imgsz)

    source = args.source
    if not source.isdigit():
        source_path = Path(source)
        if not source_path.exists():
            raise SystemExit(f"Prediction source not found: {source_path}")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    kwargs = {
        "source": source,
        "device": args.device,
        "imgsz": imgsz,
        "verbose": False,
        "save": args.save,
    }
    if mode != "stage2":
        kwargs["conf"] = args.conf

    results = model.predict(**kwargs)
    normalized = []
    for result in results:
        result_source = getattr(result, "path", source)
        if mode == "stage2":
            normalized.append(classification_output(result, source=result_source, weights=str(weights)))
        else:
            normalized.append(detection_output(result, source=result_source, weights=str(weights), mode=mode))

    print(json.dumps({"mode": mode, "predictions": normalized}, indent=2))


if __name__ == "__main__":
    main()
