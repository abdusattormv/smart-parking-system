#!/usr/bin/env python3
"""Export a trained YOLOv8 checkpoint to ONNX FP32 and ONNX INT8.

Outputs (placed in artifacts/models/):
  best.pt        — copy of the source checkpoint
  best.onnx      — FP32 ONNX export
  best_int8.onnx — INT8 quantized ONNX export

Usage:
  python ml/export.py --weights runs/parking/yolov8n_pklot/weights/best.pt
  python ml/export.py --weights runs/parking/yolov8s_pklot/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DEFAULT_OUTPUT_DIR = "artifacts/models"
DEFAULT_IMGSZ = 640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 to ONNX FP32 and INT8.")
    p.add_argument(
        "--weights",
        required=True,
        help="Path to best.pt checkpoint from a training run.",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for exported artifacts.",
    )
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found: {weights}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = weights.parent

    # Copy checkpoint
    dst_pt = out_dir / "best.pt"
    shutil.copy2(weights, dst_pt)
    print(f"Copied {weights} → {dst_pt}")

    model = YOLO(str(weights))

    # FP32 ONNX
    print("Exporting FP32 ONNX ...")
    fp32_result = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=20)
    fp32_src = Path(str(fp32_result))
    dst_onnx: Path | None = None
    if fp32_src.exists():
        dst_onnx = out_dir / "best.onnx"
        shutil.copy2(fp32_src, dst_onnx)
        size_mb = dst_onnx.stat().st_size / 1_048_576
        print(f"FP32 ONNX → {dst_onnx} ({size_mb:.1f} MB)")
    else:
        print(f"Warning: FP32 ONNX not found at expected path {fp32_src}")

    # INT8 ONNX
    if dst_onnx is not None and dst_onnx.exists():
        print("Exporting INT8 ONNX ...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            dst_int8 = out_dir / "best_int8.onnx"
            quantize_dynamic(
                str(dst_onnx),
                str(dst_int8),
                weight_type=QuantType.QInt8,
            )
            if weights_dir != out_dir:
                shutil.copy2(dst_int8, weights_dir / "best_int8.onnx")
            size_mb = dst_int8.stat().st_size / 1_048_576
            print(f"INT8 ONNX → {dst_int8} ({size_mb:.1f} MB)")
        except ImportError:
            print("Warning: onnxruntime quantization is unavailable; best_int8.onnx was not created.")
        except Exception as exc:
            print(f"Warning: INT8 quantization failed: {exc}")
    else:
        print("Warning: skipping INT8 ONNX because the FP32 export was not created.")

    print(f"\nArtifacts in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
