#!/usr/bin/env python3
"""Two-stage edge inference for the v3 smart parking pipeline.

Default path:
  static camera -> fixed ROIs -> crop each spot -> YOLOv8-cls -> smoothing -> JSON

Optional Stage 1:
  enable --stage1-detector to discover spot boxes with a YOLO detector instead of
  using configured fixed ROIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
DEFAULT_LOG_FORMAT = "json"
DEFAULT_STAGE2_THRESHOLD = 0.5
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEFAULT_CAMERA_PROBE_LIMIT = 10
DEFAULT_STAGE2_LOCAL_CHECKPOINT = "runs/stage2_cls/yolov8n_stage2/weights/best.pt"

DEFAULT_ROIS: Dict[str, Tuple[int, int, int, int]] = {
    "spot_1": (50, 100, 200, 250),
    "spot_2": (210, 100, 360, 250),
    "spot_3": (370, 100, 520, 250),
    "spot_4": (530, 100, 680, 250),
}


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v3 smart parking edge pipeline. Fixed ROIs are the default "
            "Stage 1 path; YOLO spot detection is optional behind --stage1-detector."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a parking image (static mode).")
    group.add_argument(
        "--camera",
        metavar="SOURCE",
        help="Camera source: numeric index like 0 or the macOS-only value 'iphone'.",
    )

    parser.add_argument(
        "--stage2-model",
        default=None,
        help="Primary Stage 2 classifier checkpoint. Defaults to config or yolov8n-cls.pt.",
    )
    parser.add_argument(
        "--stage1-model",
        default=None,
        help="Optional Stage 1 detector checkpoint used with --stage1-detector.",
    )
    parser.add_argument(
        "--stage1-detector",
        action="store_true",
        help="Use YOLO Stage 1 detection instead of fixed ROIs.",
    )
    parser.add_argument("--device", default=None, help="Inference device. Default from config.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--post", action="store_true", help="POST payloads to the backend.")
    parser.add_argument(
        "--save-annotated",
        help="Path to save an annotated output image in image mode.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--smooth-n",
        type=int,
        default=None,
        help="Temporal smoothing window in frames.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        metavar="MS",
        help="Target milliseconds between frames in camera mode.",
    )
    parser.add_argument(
        "--post-interval",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds between backend POSTs in camera mode.",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-format", choices=["csv", "json"], default=None)
    parser.add_argument(
        "--stage2-threshold",
        type=float,
        default=None,
        help="Occupied confidence threshold used by the edge classifier.",
    )
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    model_cfg = cfg.get("model", {})
    input_cfg = cfg.get("input", {})
    post_cfg = cfg.get("postprocess", {})
    log_cfg = cfg.get("logging", {})
    stage1_cfg = cfg.get("stage1", {})

    if args.stage2_model is None:
        args.stage2_model = model_cfg.get("stage2_path", STAGE2_MODEL_DEFAULT)
    if args.stage1_model is None:
        args.stage1_model = stage1_cfg.get("detector_path", STAGE1_MODEL_DEFAULT)
    if args.device is None:
        args.device = model_cfg.get("device", "mps")
    if args.smooth_n is None:
        args.smooth_n = post_cfg.get("smoothing_window", DEFAULT_SMOOTH_N)
    if args.frame_interval is None:
        args.frame_interval = input_cfg.get("frame_interval_ms", DEFAULT_FRAME_INTERVAL_MS)
    if args.post_interval is None:
        args.post_interval = input_cfg.get("post_interval_s", DEFAULT_POST_INTERVAL_S)
    if args.log_dir is None:
        args.log_dir = log_cfg.get("output_dir", str(DEFAULT_LOG_DIR))
    if args.log_format is None:
        args.log_format = log_cfg.get("format", DEFAULT_LOG_FORMAT)
    if args.stage2_threshold is None:
        args.stage2_threshold = post_cfg.get(
            "classifier_threshold", DEFAULT_STAGE2_THRESHOLD
        )
    configured_stage2_path = Path(str(args.stage2_model))
    if not configured_stage2_path.exists():
        local_stage2_checkpoint = Path(DEFAULT_STAGE2_LOCAL_CHECKPOINT)
        if local_stage2_checkpoint.exists():
            args.stage2_model = str(local_stage2_checkpoint)
            print(
                f"Stage 2 model not found at {configured_stage2_path}; "
                f"using {local_stage2_checkpoint}."
            )
    return args


def normalize_rois(raw_rois: dict | None) -> Dict[str, Tuple[int, int, int, int]]:
    rois = raw_rois or DEFAULT_ROIS
    normalized: Dict[str, Tuple[int, int, int, int]] = {}
    for spot_id, coords in rois.items():
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(f"ROI for {spot_id!r} must be [x1, y1, x2, y2].")
        x1, y1, x2, y2 = (int(v) for v in coords)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI for {spot_id!r} has invalid bounds: {coords}")
        normalized[str(spot_id)] = (x1, y1, x2, y2)
    return normalized


def load_rois(config_path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    cfg = load_config(config_path)
    return normalize_rois(cfg.get("rois"))


class SmoothingBuffer:
    """Per-spot majority-vote smoothing over string statuses."""

    def __init__(self, window: int = DEFAULT_SMOOTH_N) -> None:
        self._window = max(1, int(window))
        self._history: Dict[str, deque[str]] = {}

    def update(self, statuses: Dict[str, str]) -> None:
        for spot_id, status in statuses.items():
            history = self._history.setdefault(spot_id, deque(maxlen=self._window))
            history.append(status)

    def get_status(self) -> Dict[str, str]:
        smoothed: Dict[str, str] = {}
        for spot_id, history in self._history.items():
            occupied_count = sum(1 for status in history if status == "occupied")
            smoothed[spot_id] = (
                "occupied" if occupied_count > len(history) / 2 else "free"
            )
        return smoothed

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()


def get_spot_boxes(
    frame: np.ndarray,
    fixed_rois: Dict[str, Tuple[int, int, int, int]],
    stage1_model: Optional[YOLO],
    device: str,
    use_stage1_detector: bool,
) -> Dict[str, Tuple[int, int, int, int]]:
    if not use_stage1_detector:
        return fixed_rois
    if stage1_model is None:
        raise ValueError("Stage 1 detector requested but no model was loaded.")

    results = stage1_model(frame, device=device, verbose=False)[0]
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for index, box in enumerate(results.boxes.xyxy.cpu().numpy(), start=1):
        x1, y1, x2, y2 = box.astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes[f"spot_{index}"] = (int(x1), int(y1), int(x2), int(y2))
    return boxes


def clip_box(
    frame_shape: Tuple[int, int, int],
    box: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_patch(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    clipped = clip_box(frame.shape, box)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return patch


def _result_label_confidence(result: Any) -> Tuple[str, float]:
    top1 = int(result.probs.top1)
    label = str(result.names[top1])
    confidence = float(result.probs.top1conf)
    return label, confidence


def classify_patch(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    model: YOLO,
    device: str,
    threshold: float,
) -> Tuple[str, float]:
    patch = crop_patch(frame, box)
    if patch is None:
        return "free", 0.0
    resized = cv2.resize(patch, (64, 64))
    result = model(resized, device=device, verbose=False)[0]
    label, confidence = _result_label_confidence(result)
    status = "occupied" if label == "occupied" and confidence >= threshold else "free"
    return status, confidence


def build_payload(
    statuses: Dict[str, str],
    confidences: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "spots": dict(sorted(statuses.items())),
        "confidence": {k: round(v, 3) for k, v in sorted(confidences.items())},
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def annotate_frame(
    frame: np.ndarray,
    statuses: Dict[str, str],
    spot_boxes: Dict[str, Tuple[int, int, int, int]],
    confidences: Dict[str, float],
) -> np.ndarray:
    annotated = frame.copy()
    for spot_id, status in statuses.items():
        box = clip_box(frame.shape, spot_boxes.get(spot_id, (0, 0, 0, 0)))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = (255, 0, 255) if status == "occupied" else (47, 255, 173)
        text_color = (255, 255, 255) if status == "occupied" else (0, 0, 0)
        confidence = confidences.get(spot_id, 0.0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        label = f"{int(confidence * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tx, ty = x1, y1 -6
        cv2.rectangle(annotated, (tx - 1, ty - th - 1), (tx + tw + 1, ty + baseline), color, -1)
        cv2.putText(annotated, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return annotated


def log_result(payload: Dict[str, Any], log_dir: Path, log_format: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"parking_log_{datetime.now().strftime('%Y-%m-%d')}.{log_format}"
    if log_format == "json":
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return

    fieldnames = ["timestamp", *payload["spots"].keys()]
    write_header = not log_path.exists()
    row = {"timestamp": payload["timestamp"], **payload["spots"]}
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def post_payload(payload: Dict[str, Any], backend_url: str) -> None:
    response = requests.post(backend_url, json=payload, timeout=5)
    response.raise_for_status()


def run_pipeline(
    frame: np.ndarray,
    fixed_rois: Dict[str, Tuple[int, int, int, int]],
    stage1_model: Optional[YOLO],
    stage2_model: YOLO,
    smooth_buf: SmoothingBuffer,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, int, int, int]]]:
    spot_boxes = get_spot_boxes(
        frame=frame,
        fixed_rois=fixed_rois,
        stage1_model=stage1_model,
        device=args.device,
        use_stage1_detector=args.stage1_detector,
    )

    raw_statuses: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    for spot_id, box in sorted(spot_boxes.items()):
        status, confidence = classify_patch(
            frame=frame,
            box=box,
            model=stage2_model,
            device=args.device,
            threshold=args.stage2_threshold,
        )
        raw_statuses[spot_id] = status
        confidences[spot_id] = confidence

    smooth_buf.update(raw_statuses)
    payload = build_payload(smooth_buf.get_status(), confidences)
    return payload, spot_boxes


def create_models(args: argparse.Namespace) -> Tuple[Optional[YOLO], YOLO]:
    stage1_model = YOLO(args.stage1_model) if args.stage1_detector else None
    stage2_model = YOLO(args.stage2_model)
    return stage1_model, stage2_model


def _collect_camera_names(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if lowered in {"_name", "name", "spcamera_name"} and isinstance(value, str):
                yield value
            yield from _collect_camera_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from _collect_camera_names(item)


def _has_iphone_camera(camera_data: dict) -> bool:
    keywords = ("iphone", "continuity")
    for name in _collect_camera_names(camera_data):
        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            return True
    return False


def _macos_camera_backend() -> int:
    return int(getattr(cv2, "CAP_AVFOUNDATION", 0))


def _probe_camera_index(index: int, backend: Optional[int] = None) -> bool:
    if backend in (None, 0):
        capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index, backend)
    try:
        if not capture.isOpened():
            return False
        ok, _frame = capture.read()
        return bool(ok)
    finally:
        capture.release()


def resolve_camera_source(source: str, probe_limit: int = DEFAULT_CAMERA_PROBE_LIMIT) -> int:
    value = str(source).strip()
    if value.isdigit():
        return int(value)
    if value.lower() != "iphone":
        raise ValueError("Invalid --camera value. Use a numeric index like '0' or 'iphone'.")
    if platform.system() != "Darwin":
        raise ValueError("The iPhone camera option is supported only on macOS.")

    result = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True,
        text=True,
        check=True,
    )
    camera_data = json.loads(result.stdout or "{}")
    if not _has_iphone_camera(camera_data):
        raise ValueError(
            "No iPhone camera detected. Connect or enable Continuity Camera in macOS and try again."
        )

    backend = _macos_camera_backend()
    for index in range(max(1, int(probe_limit))):
        if _probe_camera_index(index, backend):
            return index
    raise RuntimeError("Detected an iPhone camera but could not open any AVFoundation camera index.")


def run_inference(args: argparse.Namespace, fixed_rois: Dict[str, Tuple[int, int, int, int]]) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    stage1_model, stage2_model = create_models(args)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)
    payload, spot_boxes = run_pipeline(frame, fixed_rois, stage1_model, stage2_model, smooth_buf, args)

    print(json.dumps(payload, indent=2))
    log_result(payload, Path(args.log_dir), args.log_format)

    if args.save_annotated:
        output_path = Path(args.save_annotated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotate_frame(frame, payload["spots"], spot_boxes, payload["confidence"]))

    if args.post:
        try:
            post_payload(payload, args.backend_url)
        except requests.RequestException as exc:
            print(f"Backend POST failed: {exc}")
    return 0


def run_camera(args: argparse.Namespace, fixed_rois: Dict[str, Tuple[int, int, int, int]]) -> int:
    stage1_model, stage2_model = create_models(args)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    last_post = time.perf_counter() - args.post_interval
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            started_at = time.perf_counter()
            payload, spot_boxes = run_pipeline(frame, fixed_rois, stage1_model, stage2_model, smooth_buf, args)

            if time.perf_counter() - last_post >= args.post_interval:
                annotated = annotate_frame(frame, payload["spots"], spot_boxes, payload["confidence"])
                cv2.imwrite(f"logs/annotated_{payload['timestamp'].replace(':', '-')}.jpg", annotated)

                print(json.dumps(payload))
                log_result(payload, Path(args.log_dir), args.log_format)
                if args.post:
                    try:
                        post_payload(payload, args.backend_url)
                    except requests.RequestException as exc:
                        print(f"Backend POST failed: {exc}")
                last_post = time.perf_counter()

            sleep_seconds = max(0.0, args.frame_interval / 1000.0 - (time.perf_counter() - started_at))
            if sleep_seconds:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
    return 0


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    args = resolve_settings(args, cfg)
    fixed_rois = normalize_rois(cfg.get("rois"))

    if args.camera is not None:
        args.camera = resolve_camera_source(args.camera)
        raise SystemExit(run_camera(args, fixed_rois))
    raise SystemExit(run_inference(args, fixed_rois))


if __name__ == "__main__":
    main()