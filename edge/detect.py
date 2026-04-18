#!/usr/bin/env python3
"""Two-stage edge inference pipeline for smart parking detection.

Stage 1 — Spot location:
  Option A (default): Fixed ROI bounding boxes defined in FIXED_ROIS.
  Option B (--no-fixed-roi): YOLO spot detector trained on Roboflow dataset.

Stage 2 — Patch classification:
  YOLOv8-cls model classifies each cropped spot patch as occupied or free.
  Default model: yolov8n-cls.pt (pre-trained placeholder until trained model ready).

Usage:
  python edge/detect.py --image /path/to/parking.jpg
  python edge/detect.py --image /path/to/parking.jpg --post
  python edge/detect.py --image /path/to/parking.jpg --save-annotated logs/out.jpg
  python edge/detect.py --image /path/to/parking.jpg --device cpu
  python edge/detect.py --image /path/to/parking.jpg --stage2-model stage2_cls/weights/best.pt
  python edge/detect.py --camera 0
"""

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import yaml
from ultralytics import YOLO

STAGE2_MODEL_DEFAULT = "yolov8n-cls.pt"
STAGE1_MODEL_DEFAULT = "yolov8n.pt"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/update"
DEFAULT_SMOOTH_N = 5
DEFAULT_FRAME_INTERVAL_MS = 250
DEFAULT_POST_INTERVAL_S = 2.0
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FORMAT = "csv"

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Fixed ROI bounding boxes: (x1, y1, x2, y2) pixel coordinates.
# Adjust these once using an image viewer to match your camera angle.
FIXED_ROIS: Dict[str, Tuple[int, int, int, int]] = {
    "spot_1": (50,  100, 200, 250),
    "spot_2": (210, 100, 360, 250),
    "spot_3": (370, 100, 520, 250),
    "spot_4": (530, 100, 680, 250),
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage smart parking edge inference pipeline."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", help="Path to a parking image (static mode).")
    group.add_argument("--camera", type=int, metavar="INDEX",
                       help="Camera device index for live capture mode.")

    parser.add_argument("--stage2-model", default=None,
                        help="Stage 2 classifier model path. Default: yolov8n-cls.pt.")
    parser.add_argument("--stage1-model", default=None,
                        help="Stage 1 spot detector model path (used with --no-fixed-roi).")
    parser.add_argument("--no-fixed-roi", action="store_true",
                        help="Use Stage 1 YOLO spot detector instead of fixed ROIs.")
    parser.add_argument("--device", default=None,
                        help="Inference device: mps or cpu. Default: mps.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--post", action="store_true",
                        help="POST each payload to the backend.")
    parser.add_argument("--save-annotated",
                        help="Path to save annotated output image (image mode only).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smooth-n", type=int, default=None,
                        help="Temporal smoothing window in frames.")
    parser.add_argument("--frame-interval", type=int, default=None, metavar="MS",
                        help="Target ms between frames in camera mode.")
    parser.add_argument("--post-interval", type=float, default=None, metavar="SEC",
                        help="Seconds between backend POSTs in camera mode.")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-format", choices=["csv", "json"], default=None)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    model_cfg = cfg.get("model", {})
    input_cfg = cfg.get("input", {})
    post_cfg  = cfg.get("postprocess", {})
    log_cfg   = cfg.get("logging", {})

    if args.stage2_model is None:
        args.stage2_model = model_cfg.get("stage2_path", STAGE2_MODEL_DEFAULT)
    if args.stage1_model is None:
        args.stage1_model = model_cfg.get("stage1_path", STAGE1_MODEL_DEFAULT)
    if args.device is None:
        args.device = model_cfg.get("device", "mps")
    if args.smooth_n is None:
        args.smooth_n = post_cfg.get("smoothing_window", DEFAULT_SMOOTH_N)
    if args.frame_interval is None:
        args.frame_interval = input_cfg.get("frame_interval_ms", DEFAULT_FRAME_INTERVAL_MS)
    if args.post_interval is None:
        args.post_interval = DEFAULT_POST_INTERVAL_S
    if args.log_dir is None:
        args.log_dir = log_cfg.get("output_dir", str(DEFAULT_LOG_DIR))
    if args.log_format is None:
        args.log_format = log_cfg.get("format", DEFAULT_LOG_FORMAT)
    return args


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

class SmoothingBuffer:
    """Per-spot majority-vote smoother. Ties resolve to free (safer default)."""

    def __init__(self, window: int = DEFAULT_SMOOTH_N) -> None:
        self._window = window
        self._history: Dict[str, deque] = {}

    def update(self, statuses: Dict[str, str]) -> None:
        for sid, status in statuses.items():
            if sid not in self._history:
                self._history[sid] = deque(maxlen=self._window)
            self._history[sid].append(status == "occupied")

    def get_status(self) -> Dict[str, str]:
        return {
            sid: "occupied" if (hist and sum(hist) > len(hist) / 2) else "free"
            for sid, hist in self._history.items()
        }

    def reset(self) -> None:
        for hist in self._history.values():
            hist.clear()


# ---------------------------------------------------------------------------
# Two-stage inference
# ---------------------------------------------------------------------------

def get_spot_boxes(
    frame: np.ndarray,
    stage1_model: Optional[YOLO],
    device: str,
    use_fixed_roi: bool,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Stage 1: return (x1, y1, x2, y2) box per spot."""
    if use_fixed_roi:
        return FIXED_ROIS
    results = stage1_model(frame, device=device, verbose=False)[0]
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        boxes[f"spot_{i + 1}"] = (int(x1), int(y1), int(x2), int(y2))
    return boxes


def classify_patch(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    model: YOLO,
    device: str,
) -> Tuple[str, float]:
    """Stage 2: crop patch, classify as occupied or free."""
    x1, y1, x2, y2 = box
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return "free", 0.0
    patch = cv2.resize(patch, (64, 64))
    result = model(patch, device=device, verbose=False)[0]
    label = result.names[result.probs.top1]
    conf = float(result.probs.top1conf)
    return label, conf


def build_payload(
    smoothed: Dict[str, str],
    confidences: Dict[str, float],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(smoothed)
    payload["confidence"] = {k: round(v, 3) for k, v in confidences.items()}
    payload["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return payload


def annotate_frame(
    frame: np.ndarray,
    smoothed: Dict[str, str],
    spot_boxes: Dict[str, Tuple[int, int, int, int]],
    confidences: Dict[str, float],
) -> np.ndarray:
    out = frame.copy()
    for spot_id, status in smoothed.items():
        box = spot_boxes.get(spot_id)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = (0, 0, 200) if status == "occupied" else (0, 180, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        conf = confidences.get(spot_id, 0.0)
        text = f"{spot_id}: {status} ({conf:.2f})"
        cv2.putText(out, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_result(payload: Dict[str, Any], log_dir: Path, log_format: str) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = log_dir / f"parking_log_{date_str}.{log_format}"

    if log_format == "json":
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    else:
        spot_keys = sorted(
            k for k in payload if k not in {"confidence", "timestamp"}
        )
        fieldnames = spot_keys + ["timestamp"]
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            row = {k: payload[k] for k in fieldnames if k in payload}
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Inference modes
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    use_fixed = not args.no_fixed_roi
    stage1 = None if use_fixed else YOLO(args.stage1_model)
    stage2 = YOLO(args.stage2_model)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)

    spot_boxes = get_spot_boxes(frame, stage1, args.device, use_fixed)

    raw_statuses: Dict[str, str] = {}
    confidences:  Dict[str, float] = {}
    for spot_id, box in spot_boxes.items():
        label, conf = classify_patch(frame, box, stage2, args.device)
        raw_statuses[spot_id] = "occupied" if label == "occupied" else "free"
        confidences[spot_id] = conf

    smooth_buf.update(raw_statuses)
    smoothed = smooth_buf.get_status()
    payload = build_payload(smoothed, confidences)

    print(json.dumps(payload, indent=2))
    log_result(payload, Path(args.log_dir), args.log_format)

    if args.save_annotated:
        out_path = Path(args.save_annotated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = annotate_frame(frame, smoothed, spot_boxes, confidences)
        cv2.imwrite(str(out_path), annotated)
        print(f"Annotated image saved to: {out_path}")

    if args.post:
        try:
            r = requests.post(args.backend_url, json=payload, timeout=5)
            print(f"Backend: {r.text}")
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"Backend POST failed: {exc}")

    return 0


def run_camera(args: argparse.Namespace) -> int:
    use_fixed = not args.no_fixed_roi
    stage1 = None if use_fixed else YOLO(args.stage1_model)
    stage2 = YOLO(args.stage2_model)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print(f"Camera {args.camera} opened. Press Ctrl-C to stop.")
    last_post = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — stopping.")
                break

            t0 = time.perf_counter()
            spot_boxes = get_spot_boxes(frame, stage1, args.device, use_fixed)

            raw_statuses: Dict[str, str] = {}
            confidences:  Dict[str, float] = {}
            for spot_id, box in spot_boxes.items():
                label, conf = classify_patch(frame, box, stage2, args.device)
                raw_statuses[spot_id] = "occupied" if label == "occupied" else "free"
                confidences[spot_id] = conf

            smooth_buf.update(raw_statuses)
            elapsed = time.perf_counter() - t0

            now = time.perf_counter()
            if now - last_post >= args.post_interval:
                smoothed = smooth_buf.get_status()
                payload = build_payload(smoothed, confidences)
                print(json.dumps(payload))
                log_result(payload, Path(args.log_dir), args.log_format)

                if args.post:
                    try:
                        requests.post(args.backend_url, json=payload, timeout=2)
                    except requests.RequestException as exc:
                        print(f"Backend POST failed: {exc}")

                last_post = now

            sleep_s = max(0.0, args.frame_interval / 1000.0 - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()

    return 0


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    args = resolve_settings(args, cfg)

    if args.camera is not None:
        raise SystemExit(run_camera(args))
    else:
        if args.image is None:
            import sys
            print("error: one of --image or --camera is required", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(run_inference(args))


if __name__ == "__main__":
    main()
