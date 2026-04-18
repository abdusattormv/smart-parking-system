#!/usr/bin/env python3
"""Prepare parking spot patch datasets for training.

── clf-data workflow (SVM + YOLOv8-cls on a simple binary dataset) ──────────
Input:
  clf-data/
    empty/      *.jpg
    not_empty/  *.jpg

Outputs: datasets/clf/  (YOLO-cls layout),  datasets/clf_svm/  (numpy arrays)

Usage:
  python ml/prepare_dataset.py --clf-dir path/to/clf-data [--skip-svm]

── stage2 workflow (PKLot + CNRPark-EXT → stage2_data/) ─────────────────────
Input:
  --pklot-dir     PKLot patch directory (Empty/ and Occupied/ subdirs anywhere)
  --cnrpark-dir   CNRPark-EXT patch directory (optional; used for cross-dataset)

Output:
  stage2_data/
    train/{occupied,free}/   PKLot train + CNRPark train (if provided)
    val/{occupied,free}/     PKLot val   + CNRPark val   (if provided)
    test/{occupied,free}/    PKLot test  + CNRPark test  (if provided)
  cnrpark_test/{occupied,free}/   CNRPark-EXT test split only (cross-dataset eval)
  pklot_test/{occupied,free}/     PKLot test split only       (cross-dataset eval)

Usage:
  python ml/prepare_dataset.py --pklot-dir datasets/pklot_patches
  python ml/prepare_dataset.py --pklot-dir datasets/pklot_patches \\
                                --cnrpark-dir datasets/cnrpark
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
        description="Prepare parking spot patches for SVM + YOLOv8-cls training."
    )
    # ── clf-data workflow ────────────────────────────────────────────────────
    p.add_argument("--clf-dir", default="datasets/clf-data",
                   help="Root dir with empty/ and not_empty/ sub-folders.")
    p.add_argument("--clf-zip", default=None,
                   help="Path to clf-data.zip; extracted to --clf-dir first.")
    p.add_argument("--output-dir", default="datasets/clf",
                   help="Output root for YOLOv8-cls dataset.")
    p.add_argument("--svm-output-dir", default="datasets/clf_svm",
                   help="Output root for SVM numpy arrays.")
    p.add_argument("--svm-imgsz",  type=int, default=SVM_IMGSZ_DEFAULT,
                   help=f"Resize NxN for SVM features (default {SVM_IMGSZ_DEFAULT}).")
    p.add_argument("--data-yaml", default="ml/data.yaml")
    p.add_argument("--skip-yolo", action="store_true",
                   help="Skip YOLOv8-cls folder preparation.")
    p.add_argument("--skip-svm",  action="store_true",
                   help="Skip SVM numpy array preparation.")
    # ── stage2 workflow ──────────────────────────────────────────────────────
    p.add_argument("--pklot-dir", default=None,
                   help="PKLot patch root (contains Empty/ and Occupied/ dirs).")
    p.add_argument("--cnrpark-dir", default=None,
                   help="CNRPark-EXT patch root (optional; for cross-dataset eval).")
    p.add_argument("--stage2-output", default="stage2_data",
                   help="Output root for combined stage2 dataset.")
    # ── shared ───────────────────────────────────────────────────────────────
    p.add_argument("--val-ratio",  type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
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
# Stage 2 — PKLot + CNRPark-EXT combined dataset
# ---------------------------------------------------------------------------

# Folder names that indicate free / occupied spots (matched case-insensitively)
_FREE_DIRS = {"empty", "free", "0"}
_OCC_DIRS  = {"occupied", "not_empty", "1"}


def collect_stage2_images(root: Path) -> dict[str, list[Path]]:
    """Walk root recursively and classify images by their immediate parent folder."""
    result: dict[str, list[Path]] = {"occupied": [], "free": []}
    exts = {".jpg", ".jpeg", ".png"}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        parent = p.parent.name.lower()
        if parent in _FREE_DIRS:
            result["free"].append(p)
        elif parent in _OCC_DIRS:
            result["occupied"].append(p)
    return result


def copy_images(
    splits: dict[str, dict[str, list[Path]]],
    dest_root: Path,
) -> None:
    """Copy split images to dest_root/{split}/{class}/."""
    for split_name, cls_paths in splits.items():
        for cls, paths in cls_paths.items():
            dest = dest_root / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in paths:
                target = dest / src.name
                # Avoid collisions: prepend a short hash of the source path
                if target.exists():
                    h = format(hash(str(src)) & 0xFFFFFF, "06x")
                    target = dest / f"{h}_{src.name}"
                shutil.copy2(src, target)


def prepare_stage2(args: argparse.Namespace) -> None:
    pklot_dir  = Path(args.pklot_dir)
    out_dir    = Path(args.stage2_output)

    if not pklot_dir.exists():
        raise SystemExit(f"PKLot directory not found: {pklot_dir}")

    print(f"\n[Stage 2] Collecting PKLot patches from {pklot_dir} ...")
    pklot_images = collect_stage2_images(pklot_dir)
    for cls, imgs in pklot_images.items():
        print(f"  PKLot {cls}: {len(imgs)}")
    if not any(pklot_images.values()):
        raise SystemExit(
            "No images found. Ensure PKLot has Empty/ and Occupied/ subdirectories."
        )

    pklot_splits = stratified_split(
        pklot_images,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    cnrpark_splits: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}
    }
    if args.cnrpark_dir:
        cnrpark_dir = Path(args.cnrpark_dir)
        if not cnrpark_dir.exists():
            raise SystemExit(f"CNRPark directory not found: {cnrpark_dir}")
        print(f"\n[Stage 2] Collecting CNRPark-EXT patches from {cnrpark_dir} ...")
        cnr_images = collect_stage2_images(cnrpark_dir)
        for cls, imgs in cnr_images.items():
            print(f"  CNRPark {cls}: {len(imgs)}")
        cnrpark_splits = stratified_split(
            cnr_images,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    # Combined stage2_data: merge PKLot + CNRPark for each split
    combined: dict[str, dict[str, list[Path]]] = {}
    for split in ("train", "val", "test"):
        combined[split] = {}
        for cls in ("occupied", "free"):
            combined[split][cls] = (
                pklot_splits[split].get(cls, [])
                + cnrpark_splits[split].get(cls, [])
            )

    print(f"\n[Stage 2] Writing combined dataset to {out_dir}/ ...")
    copy_images(combined, out_dir)

    # Separate cross-dataset test sets
    pklot_test_dir  = Path("pklot_test")
    cnrpark_test_dir = Path("cnrpark_test")

    print(f"[Stage 2] Writing PKLot test split to {pklot_test_dir}/ ...")
    copy_images(
        {"test": pklot_splits["test"]},
        pklot_test_dir,
    )

    if args.cnrpark_dir:
        print(f"[Stage 2] Writing CNRPark test split to {cnrpark_test_dir}/ ...")
        copy_images(
            {"test": cnrpark_splits["test"]},
            cnrpark_test_dir,
        )

    print("\n── Stage 2 split summary ──────────────────────")
    for split_name, cls_paths in combined.items():
        counts = {cls: len(p) for cls, p in cls_paths.items()}
        total  = sum(counts.values())
        detail = "  ".join(f"{c}={n}" for c, n in counts.items())
        print(f"  {split_name:5s}: {total:6d}  ({detail})")
    pklot_test_n = sum(len(v) for v in pklot_splits["test"].values())
    print(f"  pklot_test : {pklot_test_n:6d}  (cross-dataset eval)")
    if args.cnrpark_dir:
        cnr_test_n = sum(len(v) for v in cnrpark_splits["test"].values())
        print(f"  cnrpark_test: {cnr_test_n:6d}  (cross-dataset eval)")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Stage 2 workflow ─────────────────────────────────────────────────────
    if args.pklot_dir:
        prepare_stage2(args)
        return

    # ── clf-data workflow ────────────────────────────────────────────────────
    clf_dir    = Path(args.clf_dir)
    output_dir = Path(args.output_dir)
    svm_dir    = Path(args.svm_output_dir)

    if args.clf_zip:
        extract_zip(Path(args.clf_zip), clf_dir)

    if not clf_dir.exists():
        raise SystemExit(
            f"clf-data directory not found: {clf_dir}\n"
            "Use --clf-zip clf-data.zip, or set --clf-dir to the correct path.\n"
            "For the Stage 2 pipeline use --pklot-dir instead."
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