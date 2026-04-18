#!/usr/bin/env python3
"""Prepare parking spot datasets for training.

── Stage 1 workflow (YOLOv8 detection on full frames) ───────────────────────
Input:  Roboflow PKLot export  (train/valid/test  +  images/ + labels/)
        Labels have 2 classes: 0=empty, 1=occupied

Output: stage1_data/
          train/{images,labels}/
          valid/{images,labels}/
          test/{images,labels}/
        ml/stage1.yaml
        Labels are remapped to a single class 0 = "space" so Stage 1 only
        detects where spaces are; Stage 2 handles occupancy.

Usage:
  python ml/prepare_dataset.py --stage1 --pklot-dir datasets/pklot_raw

── Stage 2 workflow (YOLOv8-cls on cropped patches) ─────────────────────────
Input:  Roboflow PKLot export  (same dir; patches are cropped from frames)
        --cnrpark-dir  CNRPark-EXT patch directory (optional)

Output: stage2_data/
          train/{occupied,free}/
          val/{occupied,free}/
          test/{occupied,free}/
        pklot_test/{occupied,free}/   (cross-dataset eval)
        cnrpark_test/{occupied,free}/ (cross-dataset eval, if --cnrpark-dir)

Usage:
  python ml/prepare_dataset.py --stage2 --pklot-dir datasets/pklot_raw
  python ml/prepare_dataset.py --stage2 --pklot-dir datasets/pklot_raw \\
                                --cnrpark-dir datasets/cnrpark

Run both at once:
  python ml/prepare_dataset.py --stage1 --stage2 --pklot-dir datasets/pklot_raw
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

# Folder names that indicate free / occupied spots (matched case-insensitively)
_FREE_DIRS = {"empty", "free", "0"}
_OCC_DIRS  = {"occupied", "not_empty", "1"}

# Roboflow PKLot class IDs.
# If Stage 2 accuracy looks inverted (~0%), swap these two sets.
_ROBOFLOW_FREE_IDS = {0}   # "empty"
_ROBOFLOW_OCC_IDS  = {1}   # "occupied"

STAGE1_DATA_DIR = "stage1_data"
STAGE1_YAML     = "ml/stage1.yaml"
STAGE2_DATA_DIR = "stage2_data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare PKLot for Stage 1 (detection) and/or Stage 2 (classification)."
    )
    p.add_argument("--pklot-dir", required=True,
                   help="PKLot Roboflow root (contains train/valid/test subdirs).")
    p.add_argument("--cnrpark-dir", default=None,
                   help="CNRPark-EXT patch root (optional; Stage 2 only).")
    # ── workflow selectors ───────────────────────────────────────────────────
    p.add_argument("--stage1", action="store_true",
                   help="Prepare Stage 1 detection dataset.")
    p.add_argument("--stage2", action="store_true",
                   help="Prepare Stage 2 classification dataset.")
    # ── output locations ─────────────────────────────────────────────────────
    p.add_argument("--stage1-output", default=STAGE1_DATA_DIR)
    p.add_argument("--stage1-yaml",   default=STAGE1_YAML)
    p.add_argument("--stage2-output", default=STAGE2_DATA_DIR)
    p.add_argument("--patch-cache",   default=None,
                   help="Where to write cropped Stage 2 patches. "
                        "Default: <pklot-dir>_patches/")
    # ── split ratios (Stage 2 re-splits patches; Stage 1 keeps Roboflow splits)
    p.add_argument("--val-ratio",  type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _roboflow_splits(root: Path) -> list[tuple[str, Path, Path]]:
    """Return [(split_name, img_dir, label_dir), ...] for all present splits."""
    result = []
    for split in ("train", "valid", "test"):
        img_dir   = root / split / "images"
        label_dir = root / split / "labels"
        # Roboflow sometimes uses "valid", sometimes "val"
        if not img_dir.exists() and split == "valid":
            img_dir   = root / "val" / "images"
            label_dir = root / "val" / "labels"
            split = "val"
        if img_dir.exists() and label_dir.exists():
            result.append((split, img_dir, label_dir))
        else:
            print(f"  [warn] skipping missing split: {split}")
    return result


def _safe_name(src: Path, source_root: Path | None) -> str:
    """Build a collision-free filename by prefixing with relative path segments."""
    if source_root is not None:
        try:
            rel = src.relative_to(source_root)
            prefix_parts = rel.parts[:-2]   # drop <class>/<filename>
            if prefix_parts:
                return "_".join(prefix_parts) + "__" + src.name
        except ValueError:
            pass
    return src.name


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


def copy_images(
    splits: dict[str, dict[str, list[Path]]],
    dest_root: Path,
    *,
    source_root: Path | None = None,
) -> None:
    """Copy split images to dest_root/{split}/{class}/."""
    for split_name, cls_paths in splits.items():
        for cls, paths in cls_paths.items():
            dest = dest_root / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in paths:
                safe   = _safe_name(src, source_root)
                target = dest / safe
                if target.exists():
                    h = format(abs(hash(str(src))) & 0xFFFFFF, "06x")
                    target = dest / f"{h}_{safe}"
                shutil.copy2(src, target)


def _copy_test_flat(
    test_cls_paths: dict[str, list[Path]],
    dest_root: Path,
    source_root: Path | None = None,
) -> None:
    """Copy test patches to dest_root/{class}/ (no extra 'test/' subdir)."""
    for cls, paths in test_cls_paths.items():
        dest = dest_root / cls
        dest.mkdir(parents=True, exist_ok=True)
        for src in paths:
            safe   = _safe_name(src, source_root)
            target = dest / safe
            if target.exists():
                h = format(abs(hash(str(src))) & 0xFFFFFF, "06x")
                target = dest / f"{h}_{safe}"
            shutil.copy2(src, target)


# ---------------------------------------------------------------------------
# Stage 1 — remap labels and copy full frames
# ---------------------------------------------------------------------------

def prepare_stage1(pklot_dir: Path, out_dir: Path, yaml_path: Path) -> None:
    """
    Copy full-frame images and remap all class IDs → 0 ("space").
    Keeps Roboflow's train/valid/test split as-is.
    """
    print(f"\n[Stage 1] Building detection dataset → {out_dir}/")

    splits = _roboflow_splits(pklot_dir)
    if not splits:
        raise SystemExit(
            f"No Roboflow splits found in {pklot_dir}.\n"
            "Expected: train/images/, valid/images/, test/images/"
        )

    total_images = 0
    total_boxes  = 0

    for split_name, img_dir, label_dir in splits:
        out_img = out_dir / split_name / "images"
        out_lbl = out_dir / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n_img = n_box = 0

        for img_path in img_files:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # Copy image unchanged
            shutil.copy2(img_path, out_img / img_path.name)

            # Remap labels: set class ID = 0 for every box, keep bbox coords
            lines = label_path.read_text().splitlines()
            remapped = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                remapped.append("0 " + " ".join(parts[1:]))
                n_box += 1

            if remapped:
                (out_lbl / (img_path.stem + ".txt")).write_text(
                    "\n".join(remapped) + "\n"
                )
                n_img += 1

        print(f"  {split_name:5s}: {n_img:5d} images  {n_box:7d} boxes")
        total_images += n_img
        total_boxes  += n_box

    # Write data YAML
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path":  str(out_dir.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["space"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Total : {total_images} images  {total_boxes} boxes")
    print(f"  YAML  → {yaml_path}")
    print(f"  Data  → {out_dir}/")


# ---------------------------------------------------------------------------
# Stage 2 — crop patches from Roboflow frames
# ---------------------------------------------------------------------------

def collect_roboflow_patches(
    root: Path,
    patch_output: Path,
) -> dict[str, list[Path]]:
    """
    Crop individual parking space patches from Roboflow-format PKLot.
    Writes patches to patch_output/{split}/{free|occupied}/*.jpg
    Returns flat {"free": [...], "occupied": [...]} of ALL patch paths.
    """
    result: dict[str, list[Path]] = {"free": [], "occupied": []}

    for split_name, img_dir, label_dir in _roboflow_splits(root):
        free_out = patch_output / split_name / "free"
        occ_out  = patch_output / split_name / "occupied"
        free_out.mkdir(parents=True, exist_ok=True)
        occ_out.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n_free = n_occ = 0

        for img_path in img_files:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img  = Image.open(img_path)
            W, H = img.size

            for i, line in enumerate(label_path.read_text().splitlines()):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = (float(x) for x in parts[1:5])

                x1 = max(0, int((cx - bw / 2) * W))
                y1 = max(0, int((cy - bh / 2) * H))
                x2 = min(W, int((cx + bw / 2) * W))
                y2 = min(H, int((cy + bh / 2) * H))
                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img.crop((x1, y1, x2, y2))
                fname = f"{img_path.stem}__{i:04d}.jpg"

                if cls_id in _ROBOFLOW_FREE_IDS:
                    patch.save(free_out / fname)
                    result["free"].append(free_out / fname)
                    n_free += 1
                elif cls_id in _ROBOFLOW_OCC_IDS:
                    patch.save(occ_out / fname)
                    result["occupied"].append(occ_out / fname)
                    n_occ += 1

        print(f"  {split_name:5s}: {n_free:7d} free  {n_occ:7d} occupied patches")

    return result


def collect_legacy_patches(root: Path) -> dict[str, list[Path]]:
    """Walk root for Empty/ and Occupied/ leaf dirs (CNRPark-EXT layout)."""
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


def prepare_stage2(args: argparse.Namespace) -> None:
    pklot_dir = Path(args.pklot_dir)
    out_dir   = Path(args.stage2_output)

    # ── Crop PKLot patches ───────────────────────────────────────────────────
    patch_cache = Path(args.patch_cache) if args.patch_cache else (
        pklot_dir.parent / (pklot_dir.name + "_patches")
    )
    print(f"\n[Stage 2] Cropping PKLot patches → {patch_cache}/")
    pklot_images = collect_roboflow_patches(pklot_dir, patch_cache)

    for cls, imgs in pklot_images.items():
        print(f"  PKLot total {cls}: {len(imgs)}")
    if not any(pklot_images.values()):
        raise SystemExit("No patches cropped from PKLot. Check --pklot-dir.")

    # ── CNRPark-EXT (optional) ───────────────────────────────────────────────
    cnrpark_splits: dict[str, dict[str, list[Path]]] = {
        "train": {"occupied": [], "free": []},
        "val":   {"occupied": [], "free": []},
        "test":  {"occupied": [], "free": []},
    }
    cnr_source_root: Path | None = None

    if args.cnrpark_dir:
        cnrpark_dir = Path(args.cnrpark_dir)
        if not cnrpark_dir.exists():
            raise SystemExit(f"CNRPark directory not found: {cnrpark_dir}")
        print(f"\n[Stage 2] Collecting CNRPark-EXT patches from {cnrpark_dir} ...")
        cnr_images = collect_legacy_patches(cnrpark_dir)
        for cls, imgs in cnr_images.items():
            print(f"  CNRPark {cls}: {len(imgs)}")
        cnrpark_splits = stratified_split(
            cnr_images,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        cnr_source_root = cnrpark_dir

    # ── Split PKLot patches ──────────────────────────────────────────────────
    pklot_splits = stratified_split(
        pklot_images,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ── Merge and write stage2_data/ ─────────────────────────────────────────
    combined: dict[str, dict[str, list[Path]]] = {}
    for split in ("train", "val", "test"):
        combined[split] = {}
        for cls in ("occupied", "free"):
            combined[split][cls] = (
                pklot_splits[split].get(cls, [])
                + cnrpark_splits[split].get(cls, [])
            )

    print(f"\n[Stage 2] Writing combined dataset → {out_dir}/")
    copy_images(combined, out_dir, source_root=patch_cache)

    # ── Cross-dataset test splits ────────────────────────────────────────────
    pklot_test_dir   = Path("pklot_test")
    cnrpark_test_dir = Path("cnrpark_test")

    print(f"[Stage 2] Writing PKLot test split → {pklot_test_dir}/")
    _copy_test_flat(pklot_splits["test"], pklot_test_dir, source_root=patch_cache)

    if args.cnrpark_dir:
        print(f"[Stage 2] Writing CNRPark test split → {cnrpark_test_dir}/")
        _copy_test_flat(
            cnrpark_splits["test"], cnrpark_test_dir, source_root=cnr_source_root
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Stage 2 split summary ──────────────────────────")
    for split_name, cls_paths in combined.items():
        counts = {cls: len(p) for cls, p in cls_paths.items()}
        total  = sum(counts.values())
        detail = "  ".join(f"{c}={n}" for c, n in counts.items())
        print(f"  {split_name:5s}: {total:7d}  ({detail})")
    pklot_test_n = sum(len(v) for v in pklot_splits["test"].values())
    print(f"  pklot_test  : {pklot_test_n:7d}  (cross-dataset eval)")
    if args.cnrpark_dir:
        cnr_test_n = sum(len(v) for v in cnrpark_splits["test"].values())
        print(f"  cnrpark_test: {cnr_test_n:7d}  (cross-dataset eval)")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.stage1 and not args.stage2:
        raise SystemExit(
            "Specify at least one workflow:\n"
            "  --stage1              build detection dataset\n"
            "  --stage2              build classification dataset\n"
            "  --stage1 --stage2     build both"
        )

    pklot_dir = Path(args.pklot_dir)
    if not pklot_dir.exists():
        raise SystemExit(f"PKLot directory not found: {pklot_dir}")

    if not any((pklot_dir / s / "images").exists()
               for s in ("train", "valid", "test", "val")):
        raise SystemExit(
            f"No Roboflow layout detected in {pklot_dir}.\n"
            "Expected: train/images/, valid/images/, test/images/"
        )

    if args.stage1:
        prepare_stage1(
            pklot_dir,
            Path(args.stage1_output),
            Path(args.stage1_yaml),
        )

    if args.stage2:
        prepare_stage2(args)


if __name__ == "__main__":
    main()