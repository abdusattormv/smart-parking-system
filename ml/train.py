#!/usr/bin/env python3
"""Train YOLOv8 for the smart parking system.

Stage 1 — YOLOv8n detection (finds parking spaces in full camera frames)
Stage 2 — YOLOv8n-cls classification (occupied vs free on cropped patches)

Usage:
  python ml/train.py --stage1 --device cuda        # train detector
  python ml/train.py --stage2 --device cuda        # train classifier
  python ml/train.py --stage1 --variant s          # larger model
  python ml/train.py --stage1 --epochs 100         # more epochs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
import ultralytics
from ultralytics import YOLO

# ── defaults ─────────────────────────────────────────────────────────────────
STAGE1_YAML         = "ml/stage1.yaml"
STAGE2_DATA_DIR     = "stage2_data"

STAGE1_EPOCHS       = 50
STAGE2_EPOCHS       = 20

STAGE1_IMGSZ        = 640   # detection — full frames need higher resolution
STAGE2_IMGSZ        = 64    # classification — small patches

DEFAULT_BATCH       = 16    # safe default; use --batch to override
STAGE2_BATCH        = 64
DEFAULT_LR          = 0.01
DEFAULT_PATIENCE    = 10

STAGE1_PROJECT      = "runs/stage1_det"
STAGE2_PROJECT      = "runs/stage2_cls"
MODEL_DIR           = "models"

MIN_ULTRALYTICS     = (8, 4, 38)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_version() -> None:
    installed = tuple(int(x) for x in ultralytics.__version__.split(".")[:3])
    if installed < MIN_ULTRALYTICS:
        v = ".".join(str(x) for x in MIN_ULTRALYTICS)
        print(
            f"[warn] ultralytics {ultralytics.__version__} detected; "
            f"{v}+ recommended.\n"
            f"       pip install -U ultralytics",
            file=sys.stderr,
        )


def _train_with_fallback(model: YOLO, kwargs: dict, args: argparse.Namespace):
    """Run training; halve batch on MPS shape-mismatch error and retry once."""
    try:
        return model.train(**kwargs)
    except RuntimeError as exc:
        mps_crash = (
            args.device == "mps"
            and not getattr(args, "no_batch_fallback", False)
            and "shape mismatch" in str(exc)
        )
        if not mps_crash:
            raise
        new_batch = kwargs["batch"] // 2
        print(
            f"\n[MPS fallback] shape-mismatch → retrying with batch={new_batch}",
            file=sys.stderr,
        )
        kwargs = {**kwargs, "batch": new_batch, "exist_ok": True}
        return YOLO(model.ckpt_path).train(**kwargs)


def _resolve_stage1_data(args: argparse.Namespace) -> str:
    yaml_path = Path(args.data or STAGE1_YAML)
    if not yaml_path.exists():
        raise SystemExit(
            f"Stage 1 YAML not found: {yaml_path}\n"
            "Run: python ml/prepare_dataset.py --stage1 --pklot-dir datasets/pklot_raw"
        )
    return str(yaml_path)


def _resolve_stage2_data(args: argparse.Namespace) -> str:
    data_path = Path(args.data or STAGE2_DATA_DIR)
    if not data_path.exists():
        raise SystemExit(
            f"Stage 2 data not found: {data_path}\n"
            "Run: python ml/prepare_dataset.py --stage2 --pklot-dir datasets/pklot_raw"
        )
    return str(data_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8 Stage 1 (detection) or Stage 2 (classification)."
    )
    # ── workflow ──────────────────────────────────────────────────────────────
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage1", action="store_true",
                   help="Train Stage 1 detector (YOLOv8n).")
    g.add_argument("--stage2", action="store_true",
                   help="Train Stage 2 classifier (YOLOv8n-cls).")
    # ── model / data ──────────────────────────────────────────────────────────
    p.add_argument("--variant", choices=["n", "s", "m"], default="n",
                   help="Model size: n=nano, s=small, m=medium. Default: n")
    p.add_argument("--data", default=None,
                   help="Override data path (YAML for stage1, dir for stage2).")
    # ── hyperparams ───────────────────────────────────────────────────────────
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--imgsz",    type=int,   default=None,
                   help="Image size. Default: 640 (stage1), 64 (stage2).")
    p.add_argument("--batch",    type=int,   default=None,
                   help="Batch size. Default: 16 (stage1), 64 (stage2).")
    p.add_argument("--lr",       type=float, default=DEFAULT_LR, dest="lr0")
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE)
    # ── hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--device",   default="cuda",
                   help="Device: cuda | cpu | mps | 0. Default: cuda")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--no-batch-fallback", action="store_true")
    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--model-dir", default=MODEL_DIR)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _check_version()

    if args.stage1:
        data_path   = _resolve_stage1_data(args)
        weights     = f"yolov8{args.variant}.pt"          # detection weights
        epochs      = args.epochs  or STAGE1_EPOCHS
        imgsz       = args.imgsz   or STAGE1_IMGSZ
        batch       = args.batch   or DEFAULT_BATCH
        project_dir = STAGE1_PROJECT
        run_name    = f"yolov8{args.variant}_stage1"
        report_name = "stage1_report.json"
        task        = "detect"
    else:
        data_path   = _resolve_stage2_data(args)
        weights     = f"yolov8{args.variant}-cls.pt"      # classification weights
        epochs      = args.epochs  or STAGE2_EPOCHS
        imgsz       = args.imgsz   or STAGE2_IMGSZ
        batch       = args.batch   or STAGE2_BATCH
        project_dir = STAGE2_PROJECT
        run_name    = f"yolov8{args.variant}_stage2"
        report_name = "stage2_report.json"
        task        = "classify"

    print(f"Task   : {task}")
    print(f"Model  : {weights}")
    print(f"Data   : {data_path}")
    print(f"Device : {args.device}")
    print(f"Epochs : {epochs}  imgsz={imgsz}  batch={batch}\n")

    model = YOLO(weights)

    train_kwargs = dict(
        data          = data_path,
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        optimizer     = "AdamW",
        lr0           = args.lr0,
        patience      = args.patience,
        device        = args.device,
        project       = project_dir,
        name          = run_name,
        resume        = args.resume,
        verbose       = True,
        deterministic = False,
    )

    t0      = time.perf_counter()
    results = _train_with_fallback(model, train_kwargs, args)
    elapsed = time.perf_counter() - t0

    best_pt = Path(project_dir) / run_name / "weights" / "best.pt"
    print(f"\nTraining complete  ({elapsed:.0f}s)")
    print(f"Best checkpoint : {best_pt}")

    # ── metrics ───────────────────────────────────────────────────────────────
    rd = results.results_dict
    report: dict = {
        "task":         task,
        "model":        weights,
        "variant":      args.variant,
        "epochs":       epochs,
        "imgsz":        imgsz,
        "batch":        batch,
        "train_time_s": round(elapsed, 2),
        "best_ckpt":    str(best_pt),
    }

    if task == "classify":
        top1 = rd.get("metrics/accuracy_top1", rd.get("val/acc_top1"))
        top5 = rd.get("metrics/accuracy_top5", rd.get("val/acc_top5"))
        print(f"Top-1 accuracy : {top1:.4f}" if isinstance(top1, float) else f"Top-1: {top1}")
        print(f"Top-5 accuracy : {top5:.4f}" if isinstance(top5, float) else f"Top-5: {top5}")
        report.update({"top1_accuracy": top1, "top5_accuracy": top5})
    else:
        map50   = rd.get("metrics/mAP50(B)",   rd.get("val/map50"))
        map5095 = rd.get("metrics/mAP50-95(B)", rd.get("val/map"))
        print(f"mAP@50     : {map50:.4f}"   if isinstance(map50,   float) else f"mAP50: {map50}")
        print(f"mAP@50-95  : {map5095:.4f}" if isinstance(map5095, float) else f"mAP50-95: {map5095}")
        report.update({"mAP50": map50, "mAP50_95": map5095})

    # ── save report ───────────────────────────────────────────────────────────
    mdl_dir = Path(args.model_dir)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    report_path = mdl_dir / report_name
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()