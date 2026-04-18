#!/usr/bin/env python3
"""Prepare clf-data for both SVM and YOLOv8-cls training.

Input structure:
  clf-data/
    empty/      *.jpg   (empty parking spots)
    not_empty/  *.jpg   (occupied parking spots)

  OR a zip:  --clf-zip clf-data.zip   (extracted automatically)

Outputs
-------
1. datasets/clf/           YOLOv8-cls folder layout
     train/empty/  train/not_empty/
     val/empty/    val/not_empty/
     test/empty/   test/not_empty/

2. datasets/clf_svm/       Flat numpy arrays for SVM
     X_train.npy  y_train.npy
     X_val.npy    y_val.npy
     X_test.npy   y_test.npy
     meta.json    (class names, imgsz, split counts)

3. ml/data.yaml            Ultralytics data config

Both outputs share the same train/val/test split so comparisons are fair.

Usage:
  python ml/prepare_dataset.py
  python ml/prepare_dataset.py --clf-zip clf-data.zip
  python ml/prepare_dataset.py --clf-dir path/to/clf-data
  python ml/prepare_dataset.py --svm-imgsz 32   # larger SVM feature maps
  python ml/prepare_dataset.py --skip-svm        # YOLOv8-cls only
  python ml/prepare_dataset.py --skip-yolo       # SVM only
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import yaml

CLASSES = ["empty", "not_empty"]
SVM_IMGSZ_DEFAULT = 15   # matches the original Colab notebook (15×15 → 675 features)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare clf-data for SVM + YOLOv8-cls training."
    )
    p.add_argument("--clf-dir", default="datasets/clf-data",
                   help="Root dir with empty/ and not_empty/ sub-folders.")
    p.add_argument("--clf-zip", default=None,
                   help="Path to clf-data.zip; extracted to --clf-dir first.")
    p.add_argument("--output-dir", default="datasets/clf",
                   help="Output root for YOLOv8-cls dataset.")
    p.add_argument("--svm-output-dir", default="datasets/clf_svm",
                   help="Output root for SVM numpy arrays.")
    p.add_argument("--val-ratio",  type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--svm-imgsz",  type=int,   default=SVM_IMGSZ_DEFAULT,
                   help=f"Resize NxN for SVM features (default {SVM_IMGSZ_DEFAULT}).")
    p.add_argument("--data-yaml",  default="ml/data.yaml")
    p.add_argument("--skip-yolo",  action="store_true",
                   help="Skip YOLOv8-cls folder preparation.")
    p.add_argument("--skip-svm",   action="store_true",
                   help="Skip SVM numpy array preparation.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path} → {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # Unwrap a single top-level wrapper directory if present
    children = [c for c in dest.iterdir() if c.is_dir()]
    if len(children) == 1 and not (dest / "empty").exists():
        inner = children[0]
        for item in inner.iterdir():
            shutil.move(str(item), dest / item.name)
        inner.rmdir()


def collect_images(clf_dir: Path) -> dict[str, list[Path]]:
    missing = [c for c in CLASSES if not (clf_dir / c).is_dir()]
    if missing:
        raise SystemExit(
            f"Missing class folders in {clf_dir}: {missing}\n"
            f"Expected: {CLASSES}"
        )
    result = {}
    for cls in CLASSES:
        imgs = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in (clf_dir / cls).glob(ext)
        )
        if not imgs:
            raise SystemExit(f"No images found in {clf_dir / cls}/")
        result[cls] = imgs
    return result


def stratified_split(
    class_images: dict[str, list[Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    """Per-class stratified split → {split: {class: [paths]}}."""
    splits: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}
    }
    for cls, images in class_images.items():
        train_val, test = train_test_split(
            images, test_size=test_ratio, random_state=seed
        )
        val_adj = val_ratio / (1.0 - test_ratio)
        train, val = train_test_split(
            train_val, test_size=val_adj, random_state=seed
        )
        splits["train"][cls] = train
        splits["val"][cls]   = val
        splits["test"][cls]  = test
    return splits


# ---------------------------------------------------------------------------
# YOLOv8-cls output
# ---------------------------------------------------------------------------

def prepare_yolo(
    splits: dict[str, dict[str, list[Path]]],
    output_dir: Path,
    yaml_path: Path,
) -> None:
    print("\n[YOLOv8-cls] Copying images ...")
    for split_name, cls_paths in splits.items():
        for cls, paths in cls_paths.items():
            dest = output_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, dest / src.name)

    data = {
        "path":  str(output_dir.resolve()),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  → {output_dir}/")
    print(f"  → data.yaml: {yaml_path}")


# ---------------------------------------------------------------------------
# SVM output
# ---------------------------------------------------------------------------

def load_flat(paths: list[Path], imgsz: int) -> np.ndarray:
    """Resize each image to imgsz×imgsz and flatten → (N, features) float32."""
    return np.array(
        [resize(imread(str(p)), (imgsz, imgsz), anti_aliasing=True).flatten()
         for p in paths],
        dtype=np.float32,
    )


def prepare_svm(
    splits: dict[str, dict[str, list[Path]]],
    svm_dir: Path,
    imgsz: int,
) -> None:
    print(f"\n[SVM] Extracting {imgsz}×{imgsz} pixel features ...")
    svm_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"classes": CLASSES, "imgsz": imgsz, "counts": {}}
    rng = np.random.default_rng(42)

    for split_name, cls_paths in splits.items():
        X_parts, y_parts = [], []
        for label_idx, cls in enumerate(CLASSES):
            paths = cls_paths[cls]
            print(f"  {split_name}/{cls}: {len(paths)} images")
            X_parts.append(load_flat(paths, imgsz))
            y_parts.append(np.full(len(paths), label_idx, dtype=np.int32))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        # Shuffle so classes aren't contiguous (matters for cross-val ordering)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]

        np.save(svm_dir / f"X_{split_name}.npy", X)
        np.save(svm_dir / f"y_{split_name}.npy", y)
        meta["counts"][split_name] = int(len(y))

    with open(svm_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Feature dim per image: {X.shape[1]}")
    print(f"  → {svm_dir}/")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    clf_dir    = Path(args.clf_dir)
    output_dir = Path(args.output_dir)
    svm_dir    = Path(args.svm_output_dir)

    if args.clf_zip:
        extract_zip(Path(args.clf_zip), clf_dir)

    if not clf_dir.exists():
        raise SystemExit(
            f"clf-data directory not found: {clf_dir}\n"
            "Use --clf-zip clf-data.zip, or set --clf-dir to the correct path."
        )

    print(f"Reading images from {clf_dir} ...")
    class_images = collect_images(clf_dir)
    for cls, imgs in class_images.items():
        print(f"  {cls}: {len(imgs)} images")
    print(f"  total: {sum(len(v) for v in class_images.values())}")

    splits = stratified_split(
        class_images,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if not args.skip_yolo:
        prepare_yolo(splits, output_dir, Path(args.data_yaml))

    if not args.skip_svm:
        prepare_svm(splits, svm_dir, imgsz=args.svm_imgsz)

    print("\n── Split summary ──────────────────────")
    for split_name, cls_paths in splits.items():
        counts = {cls: len(p) for cls, p in cls_paths.items()}
        total  = sum(counts.values())
        detail = "  ".join(f"{c}={n}" for c, n in counts.items())
        print(f"  {split_name:5s}: {total:5d}  ({detail})")
    print()


if __name__ == "__main__":
    main()