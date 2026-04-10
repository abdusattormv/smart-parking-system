#!/usr/bin/env python3
"""FPS benchmark: compare MPS, CPU, ONNX FP32, and ONNX INT8 on a single image.

Usage:
    python edge/benchmark.py --image path/to/parking.jpg --model artifacts/models/best.pt

ONNX paths are inferred from the .pt path:
    best.pt       → best.onnx      (FP32)
    best.pt       → best_int8.onnx (INT8)

Skips any backend whose model file is not found.
"""
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge inference FPS benchmark.")
    parser.add_argument("--image", required=True, help="Path to test image.")
    parser.add_argument("--model", required=True, help="Path to .pt model file.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (not measured).",
    )
    return parser.parse_args()


def benchmark_yolo(
    model_path: str,
    frame: np.ndarray,
    device: str,
    iterations: int,
    warmup: int,
) -> Dict[str, Any]:
    """Benchmark ultralytics YOLO on a given device."""
    from ultralytics import YOLO  # deferred import so benchmark.py stays standalone

    model = YOLO(model_path)

    for _ in range(warmup):
        model(frame, device=device, verbose=False)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model(frame, device=device, verbose=False)
        times.append(time.perf_counter() - t0)

    avg_latency = sum(times) / len(times)
    return {
        "backend": f"YOLO-{device}",
        "fps": round(1 / avg_latency, 1),
        "latency_ms": round(avg_latency * 1000, 1),
        "model_size_mb": round(Path(model_path).stat().st_size / 1e6, 1),
    }


def benchmark_onnx(
    onnx_path: str,
    frame: np.ndarray,
    iterations: int,
    warmup: int,
) -> Optional[Dict[str, Any]]:
    """Benchmark ONNX Runtime on CPU. Returns None if the file does not exist."""
    if not Path(onnx_path).exists():
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  onnxruntime not installed — skipping {onnx_path}")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Preprocess: resize → normalize → NCHW float32
    resized = cv2.resize(frame, (640, 640))
    tensor = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, /255
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # HWC → 1CHW

    for _ in range(warmup):
        session.run(None, {input_name: tensor})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: tensor})
        times.append(time.perf_counter() - t0)

    avg_latency = sum(times) / len(times)
    return {
        "backend": None,  # caller sets label
        "fps": round(1 / avg_latency, 1),
        "latency_ms": round(avg_latency * 1000, 1),
        "model_size_mb": round(Path(onnx_path).stat().st_size / 1e6, 1),
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No benchmark results.")
        return

    header = f"{'Backend':<20} {'FPS':>8} {'Latency (ms)':>14} {'Model Size (MB)':>16}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    for r in results:
        print(
            f"{r['backend']:<20} {r['fps']:>8.1f} {r['latency_ms']:>14.1f}"
            f" {r['model_size_mb']:>16.1f}"
        )
    print()


def main() -> None:
    args = parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    onnx_fp32_path = model_path.with_suffix(".onnx")
    onnx_int8_path = model_path.with_name(model_path.stem + "_int8.onnx")

    results: List[Dict[str, Any]] = []

    for device in ["mps", "cpu"]:
        print(f"Benchmarking YOLO-{device} ({args.iterations} iters, {args.warmup} warmup)...")
        try:
            r = benchmark_yolo(
                str(model_path), frame, device, args.iterations, args.warmup
            )
            results.append(r)
        except Exception as exc:
            print(f"  Skipped YOLO-{device}: {exc}")

    for label, path in [("ONNX-fp32", onnx_fp32_path), ("ONNX-int8", onnx_int8_path)]:
        print(f"Benchmarking {label} ({args.iterations} iters, {args.warmup} warmup)...")
        r = benchmark_onnx(str(path), frame, args.iterations, args.warmup)
        if r is not None:
            r["backend"] = label
            results.append(r)
        else:
            print(f"  Skipped {label}: {path} not found")

    print_results_table(results)


if __name__ == "__main__":
    main()
