#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import requests
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/update"
DEFAULT_DEMO_SPOTS = 4
CAR_CLASS_IDS = {2, 3, 5, 7}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 4 static-image demo for Smart Parking System."
    )
    parser.add_argument("--image", required=True, help="Path to a parking image.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO model path or model name. Defaults to yolov8n.pt.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Inference device to request. Use cpu if mps is unavailable.",
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="POST target for the mock backend.",
    )
    parser.add_argument(
        "--demo-spots",
        type=int,
        default=DEFAULT_DEMO_SPOTS,
        help="How many demo spot keys to emit in the payload.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="POST the payload to the backend after inference.",
    )
    parser.add_argument(
        "--save-annotated",
        help="Optional path for a saved annotated output image.",
    )
    return parser.parse_args()


def build_demo_payload(
    detections: List[Dict[str, Any]],
    fps: float,
    demo_spots: int,
) -> Dict[str, Any]:
    car_detections = [item for item in detections if item["class_id"] in CAR_CLASS_IDS]
    occupied_spots = min(len(car_detections), demo_spots)
    payload: Dict[str, Any] = {}

    for index in range(demo_spots):
        payload[f"spot_{index + 1}"] = "occupied" if index < occupied_spots else "free"

    payload["confidence_avg"] = round(
        sum(item["confidence"] for item in detections) / len(detections), 4
    ) if detections else 0.0
    payload["fps"] = round(fps, 2)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


def run_inference(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    model = YOLO(args.model)
    started = time.perf_counter()
    results = model(frame, device=args.device, verbose=False)[0]
    elapsed = time.perf_counter() - started
    fps = 1 / elapsed if elapsed > 0 else 0.0

    detections: List[Dict[str, Any]] = []
    names = results.names
    boxes = results.boxes
    if boxes is not None:
        for xyxy, conf, cls in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            class_id = int(cls)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": round(float(conf), 4),
                    "bbox": [round(float(value), 2) for value in xyxy],
                }
            )

    payload = build_demo_payload(detections, fps, args.demo_spots)

    print("\nDetections")
    print(json.dumps(detections, indent=2))
    print("\nWeek 4 Demo Payload")
    print(json.dumps(payload, indent=2))

    if args.save_annotated:
        output_path = Path(args.save_annotated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), results.plot())
        print(f"\nAnnotated image saved to: {output_path}")

    if args.post:
        try:
            response = requests.post(args.backend_url, json=payload, timeout=5)
            print("\nBackend Response")
            print(response.text)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"\nBackend POST failed: {exc}")

    return 0


if __name__ == "__main__":
    cli_args = parse_args()
    raise SystemExit(run_inference(cli_args))
