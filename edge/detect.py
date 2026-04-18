#!/usr/bin/env python3
import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests
import yaml
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/update"
DEFAULT_DEMO_SPOTS = 4
CAR_CLASS_IDS = {2, 3, 5, 7}  # COCO vehicle classes (pretrained mode)
PKLOT_OCCUPIED_CLASS_ID = 1   # PKLot-trained model: 0=empty, 1=occupied

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEFAULT_SMOOTH_N = 5
DEFAULT_OVERLAP_THRESHOLD = 0.2
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_FRAME_INTERVAL_MS = 250
DEFAULT_POST_INTERVAL_S = 2.0
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FORMAT = "csv"

# Demo ROI polygons: corners in order (top-left, top-right, bottom-right, bottom-left).
# spot_occupied() uses roi[0] as top-left and roi[2] as bottom-right of the bounding rect.
DEMO_SPOTS: Dict[str, List[tuple]] = {
    "spot_1": [(50, 100), (200, 100), (200, 250), (50, 250)],
    "spot_2": [(210, 100), (360, 100), (360, 250), (210, 250)],
    "spot_3": [(370, 100), (520, 100), (520, 250), (370, 250)],
    "spot_4": [(530, 100), (680, 100), (680, 250), (530, 250)],
}


def load_config(config_path: Path) -> dict:
    """Load YAML config. Returns {} if file does not exist."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Parking System edge inference pipeline."
    )
    # Image / camera input (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", help="Path to a parking image (static mode).")
    group.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help="Camera device index for live capture mode.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="YOLO model path or name. Default: from config or yolov8n.pt.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (mps or cpu). Default: from config or mps.",
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="POST target URL for the backend.",
    )
    parser.add_argument(
        "--demo-spots",
        type=int,
        default=DEFAULT_DEMO_SPOTS,
        help="Number of demo spot slots (slices DEMO_SPOTS).",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="POST each payload to the backend.",
    )
    parser.add_argument(
        "--save-annotated",
        help="Path to save the annotated output image (image mode only).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        metavar="MS",
        help="Target ms between frames in camera mode.",
    )
    parser.add_argument(
        "--post-interval",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds between backend POSTs in camera mode.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for CSV/JSON inference logs.",
    )
    parser.add_argument(
        "--log-format",
        choices=["csv", "json"],
        default=None,
        help="Log file format.",
    )
    parser.add_argument(
        "--smooth-n",
        type=int,
        default=None,
        help="Temporal smoothing window (frames).",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=None,
        help="Minimum ROI overlap ratio to mark a spot occupied.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--pklot-model",
        action="store_true",
        help=(
            "Use PKLot-trained model mode: class 0=empty, class 1=occupied. "
            "Spots are classified directly by the model rather than by COCO vehicle overlap."
        ),
    )
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    """Merge config file values into args where CLI arg was not set (None)."""
    model_cfg = cfg.get("model", {})
    input_cfg = cfg.get("input", {})
    post_cfg = cfg.get("postprocess", {})
    log_cfg = cfg.get("logging", {})

    if args.model is None:
        args.model = model_cfg.get("path", DEFAULT_MODEL)
    if args.device is None:
        args.device = model_cfg.get("device", "mps")
    if args.confidence_threshold is None:
        args.confidence_threshold = model_cfg.get(
            "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
        )
    if args.smooth_n is None:
        args.smooth_n = post_cfg.get("smoothing_window", DEFAULT_SMOOTH_N)
    if args.overlap_threshold is None:
        args.overlap_threshold = post_cfg.get(
            "occupancy_overlap_threshold", DEFAULT_OVERLAP_THRESHOLD
        )
    if args.frame_interval is None:
        args.frame_interval = input_cfg.get(
            "frame_interval_ms", DEFAULT_FRAME_INTERVAL_MS
        )
    if args.post_interval is None:
        args.post_interval = DEFAULT_POST_INTERVAL_S
    if args.log_dir is None:
        args.log_dir = log_cfg.get("output_dir", str(DEFAULT_LOG_DIR))
    if args.log_format is None:
        args.log_format = log_cfg.get("format", DEFAULT_LOG_FORMAT)

    return args


def spot_occupied(
    boxes: List[List[float]],
    roi: List[tuple],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> bool:
    """Return True if any detected box overlaps the ROI by more than threshold.

    Uses area-ratio overlap: intersection_area / roi_area > threshold.
    threshold=0.0 reduces to a pure intersection-existence check.

    Args:
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        roi:   Four-corner polygon; roi[0] is top-left, roi[2] is bottom-right.
        threshold: Minimum overlap ratio (0.0–1.0).
    """
    if not boxes or not roi:
        return False

    rx1, ry1 = roi[0]
    rx2, ry2 = roi[2]
    roi_area = (rx2 - rx1) * (ry2 - ry1)
    if roi_area <= 0:
        return False

    for box in boxes:
        bx1, by1, bx2, by2 = box
        ix1 = max(bx1, rx1)
        iy1 = max(by1, ry1)
        ix2 = min(bx2, rx2)
        iy2 = min(by2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        overlap_ratio = (ix2 - ix1) * (iy2 - iy1) / roi_area
        if overlap_ratio > threshold:
            return True
    return False


class SmoothingBuffer:
    """Per-spot majority-vote temporal smoother.

    A spot flips status only when more than half of the recent N frames agree.
    Ties (exactly 50%) resolve to "free" — erring toward free is safer for
    parking (a driver finds a spot taken) than erring toward occupied (a driver
    never visits a free spot).
    """

    def __init__(self, spot_ids: List[str], window: int = DEFAULT_SMOOTH_N) -> None:
        self._history: Dict[str, deque] = {
            sid: deque(maxlen=window) for sid in spot_ids
        }

    def update(self, statuses: Dict[str, bool]) -> None:
        """Append one frame of boolean occupancy readings."""
        for sid, occupied in statuses.items():
            if sid in self._history:
                self._history[sid].append(bool(occupied))

    def get_status(self) -> Dict[str, str]:
        """Return majority-vote string status for each spot."""
        return {
            sid: "occupied" if (hist and sum(hist) > len(hist) / 2) else "free"
            for sid, hist in self._history.items()
        }

    def reset(self) -> None:
        """Clear all histories."""
        for hist in self._history.values():
            hist.clear()


def build_roi_payload(
    detections: List[Dict[str, Any]],
    spots: Dict[str, List[tuple]],
    smooth_buf: SmoothingBuffer,
    fps: float,
    overlap_threshold: float,
    pklot_mode: bool = False,
) -> Dict[str, Any]:
    """Build the agreed JSON payload using ROI-based spot classification.

    pklot_mode=True: model outputs occupied(1)/empty(0) spot detections directly.
    pklot_mode=False (default): COCO pretrained model; occupancy inferred from vehicle overlap.
    """
    if pklot_mode:
        occupied_boxes = [d["bbox"] for d in detections if d["class_id"] == PKLOT_OCCUPIED_CLASS_ID]
        raw: Dict[str, bool] = {
            sid: spot_occupied(occupied_boxes, roi, threshold=overlap_threshold)
            for sid, roi in spots.items()
        }
    else:
        car_boxes = [d["bbox"] for d in detections if d["class_id"] in CAR_CLASS_IDS]
        raw = {
            sid: spot_occupied(car_boxes, roi, threshold=overlap_threshold)
            for sid, roi in spots.items()
        }
    smooth_buf.update(raw)
    smoothed = smooth_buf.get_status()

    confidence_avg = (
        round(sum(d["confidence"] for d in detections) / len(detections), 4)
        if detections
        else 0.0
    )

    payload: Dict[str, Any] = dict(smoothed)
    payload["confidence_avg"] = confidence_avg
    payload["fps"] = round(fps, 2)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


def log_result(
    payload: Dict[str, Any],
    log_dir: Path,
    log_format: str,
) -> None:
    """Append one inference result to a date-stamped log file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = log_dir / f"parking_log_{date_str}.{log_format}"

    if log_format == "json":
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    else:
        # Stable field order: sorted spot keys first, then metadata columns.
        spot_keys = sorted(k for k in payload if k not in {"confidence_avg", "fps", "timestamp"})
        fieldnames = spot_keys + ["confidence_avg", "fps", "timestamp"]
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(payload)


def _active_spots(demo_spots: int) -> Dict[str, List[tuple]]:
    """Return the first N demo spots (for backward compatibility with --demo-spots)."""
    keys = list(DEMO_SPOTS.keys())[:demo_spots]
    return {k: DEMO_SPOTS[k] for k in keys}


def run_inference(args: argparse.Namespace) -> int:
    """Run single-image inference (static mode)."""
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    spots = _active_spots(args.demo_spots)
    smooth_buf = SmoothingBuffer(list(spots.keys()), window=args.smooth_n)

    model = YOLO(args.model)
    started = time.perf_counter()
    results = model(
        frame,
        device=args.device,
        conf=args.confidence_threshold,
        verbose=False,
    )[0]
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
                    "bbox": [round(float(v), 2) for v in xyxy],
                }
            )

    payload = build_roi_payload(
        detections, spots, smooth_buf, fps, args.overlap_threshold,
        pklot_mode=args.pklot_model,
    )

    print("\nDetections")
    print(json.dumps(detections, indent=2))
    print("\nPayload")
    print(json.dumps(payload, indent=2))

    log_result(payload, Path(args.log_dir), args.log_format)

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


def run_camera(args: argparse.Namespace) -> int:
    """Run live camera inference loop."""
    spots = _active_spots(args.demo_spots)
    smooth_buf = SmoothingBuffer(list(spots.keys()), window=args.smooth_n)

    model = YOLO(args.model)
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
            results = model(
                frame,
                device=args.device,
                conf=args.confidence_threshold,
                verbose=False,
            )[0]
            elapsed = time.perf_counter() - t0
            fps = 1 / elapsed if elapsed > 0 else 0.0

            detections: List[Dict[str, Any]] = []
            names = results.names
            if results.boxes is not None:
                for xyxy, conf, cls in zip(
                    results.boxes.xyxy.cpu().tolist(),
                    results.boxes.conf.cpu().tolist(),
                    results.boxes.cls.cpu().tolist(),
                ):
                    class_id = int(cls)
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": names.get(class_id, str(class_id)),
                            "confidence": round(float(conf), 4),
                            "bbox": [round(float(v), 2) for v in xyxy],
                        }
                    )

            if args.pklot_model:
                occ_boxes = [d["bbox"] for d in detections if d["class_id"] == PKLOT_OCCUPIED_CLASS_ID]
                raw: Dict[str, bool] = {
                    sid: spot_occupied(occ_boxes, roi, threshold=args.overlap_threshold)
                    for sid, roi in spots.items()
                }
            else:
                car_boxes = [d["bbox"] for d in detections if d["class_id"] in CAR_CLASS_IDS]
                raw = {
                    sid: spot_occupied(car_boxes, roi, threshold=args.overlap_threshold)
                    for sid, roi in spots.items()
                }
            smooth_buf.update(raw)

            now = time.perf_counter()
            if now - last_post >= args.post_interval:
                smoothed = smooth_buf.get_status()
                confidence_avg = (
                    round(sum(d["confidence"] for d in detections) / len(detections), 4)
                    if detections
                    else 0.0
                )
                payload: Dict[str, Any] = dict(smoothed)
                payload["confidence_avg"] = confidence_avg
                payload["fps"] = round(fps, 2)
                payload["timestamp"] = datetime.now(timezone.utc).isoformat()

                print(json.dumps(payload))
                log_result(payload, Path(args.log_dir), args.log_format)

                if args.post:
                    try:
                        requests.post(args.backend_url, json=payload, timeout=2)
                    except requests.RequestException as exc:
                        print(f"Backend POST failed: {exc}")

                last_post = now

            # Throttle to target frame interval
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
