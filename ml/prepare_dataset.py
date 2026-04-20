#!/usr/bin/env python3
"""Prepare v3 datasets for smart parking training.

Primary path:
  Stage 2 classification on cropped parking-spot patches assembled from
  PKLot Roboflow exports plus optional CNRPark-EXT patches.

Secondary path:
  Stage 1 detection dataset generation remains available, but is not the
  default success path for the repo.

ML-only baseline:
  Single-model full-frame occupancy detection keeps PKLot frame labels as
  `free` and `occupied` without a separate Stage 2 classifier.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

_FREE_DIRS = {"empty", "free", "0"}
_OCC_DIRS = {"occupied", "not_empty", "1"}
_ROBOFLOW_FREE_IDS = {0}
_ROBOFLOW_OCC_IDS = {1}

STAGE1_DATA_DIR = "stage1_data"
STAGE1_YAML = "ml/stage1.yaml"
STAGE2_DATA_DIR = "stage2_data"
SINGLE_MODEL_DATA_DIR = "single_model_data"
SINGLE_MODEL_YAML = "ml/single_model.yaml"
PKLOT_TEST_DIR = "pklot_test"
CNRPARK_TEST_DIR = "cnrpark_test"
SANITY_REPORT = "dataset_report.json"
WEATHER_CONVENTION = (
    "Expected weather layout under the dataset root: "
    "<root>/<weather>/<class>/*.jpg where weather is one of sunny, cloudy, rainy."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare smart parking datasets. Stage 2 classification is the primary "
            "workflow; Stage 1 detection support remains optional."
        )
    )
    parser.add_argument("--pklot-dir", required=True)
    parser.add_argument("--cnrpark-dir", default=None)
    parser.add_argument("--stage1", action="store_true", help="Prepare Stage 1 detection data.")
    parser.add_argument("--stage2", action="store_true", help="Prepare Stage 2 classification data.")
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Prepare ML-only single-model occupancy detection data.",
    )
    parser.add_argument("--stage1-output", default=STAGE1_DATA_DIR)
    parser.add_argument("--stage1-yaml", default=STAGE1_YAML)
    parser.add_argument("--stage2-output", default=STAGE2_DATA_DIR)
    parser.add_argument("--single-model-output", default=SINGLE_MODEL_DATA_DIR)
    parser.add_argument("--single-model-yaml", default=SINGLE_MODEL_YAML)
    parser.add_argument("--patch-cache", default=None)
    parser.add_argument("--pklot-test-output", default=PKLOT_TEST_DIR)
    parser.add_argument("--cnrpark-test-output", default=CNRPARK_TEST_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _roboflow_splits(root: Path) -> list[tuple[str, Path, Path]]:
    result = []
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.exists() and split == "valid":
            image_dir = root / "val" / "images"
            label_dir = root / "val" / "labels"
            split = "val"
        if image_dir.exists() and label_dir.exists():
            result.append((split, image_dir, label_dir))
    return result


def _safe_name(src: Path, source_root: Path | None) -> str:
    if source_root is not None:
        try:
            rel = src.relative_to(source_root)
            prefix = "_".join(rel.parts[:-2])
            if prefix:
                return f"{prefix}__{src.name}"
        except ValueError:
            pass
    return src.name


def _image_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(path.glob(suffix)))
    return files


def validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1:
        raise SystemExit("val_ratio and test_ratio must be positive and sum to less than 1.")


def stratified_split(
    class_images: dict[str, list[Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    splits: dict[str, dict[str, list[Path]]] = {"train": {}, "val": {}, "test": {}}
    for class_name, images in class_images.items():
        if len(images) < 3:
            raise SystemExit(
                f"Need at least 3 images for class {class_name!r} to create train/val/test splits."
            )
        train_val, test = train_test_split(images, test_size=test_ratio, random_state=seed)
        val_adj = val_ratio / (1.0 - test_ratio)
        train, val = train_test_split(train_val, test_size=val_adj, random_state=seed)
        splits["train"][class_name] = list(train)
        splits["val"][class_name] = list(val)
        splits["test"][class_name] = list(test)
    return splits


def copy_images(
    splits: dict[str, dict[str, list[Path]]],
    dest_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, int]:
    collision_counts: dict[str, int] = defaultdict(int)
    for split_name, class_map in splits.items():
        for class_name, paths in class_map.items():
            target_dir = dest_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                target = target_dir / _safe_name(src, source_root)
                if target.exists():
                    collision_counts[f"{split_name}/{class_name}"] += 1
                    stem, suffix = target.stem, target.suffix
                    target = target_dir / f"{stem}__{abs(hash(str(src))) & 0xFFFFFF:06x}{suffix}"
                shutil.copy2(src, target)
    return dict(collision_counts)


def copy_test_flat(
    class_map: dict[str, list[Path]],
    dest_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, int]:
    collisions: dict[str, int] = defaultdict(int)
    for class_name, paths in class_map.items():
        target_dir = dest_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            target = target_dir / _safe_name(src, source_root)
            if target.exists():
                collisions[class_name] += 1
                stem, suffix = target.stem, target.suffix
                target = target_dir / f"{stem}__{abs(hash(str(src))) & 0xFFFFFF:06x}{suffix}"
            shutil.copy2(src, target)
    return dict(collisions)


def summarize_dimensions(paths: Iterable[Path], sample_limit: int = 25) -> dict[str, object]:
    widths: list[int] = []
    heights: list[int] = []
    for path in list(paths)[:sample_limit]:
        with Image.open(path) as img:
            width, height = img.size
        widths.append(width)
        heights.append(height)
    if not widths:
        return {"sampled": 0}
    return {
        "sampled": len(widths),
        "width": {"min": min(widths), "max": max(widths)},
        "height": {"min": min(heights), "max": max(heights)},
    }


def report_duplicate_sources(class_images: dict[str, list[Path]]) -> list[str]:
    seen: Counter[str] = Counter()
    for paths in class_images.values():
        for path in paths:
            seen[str(path.resolve())] += 1
    return [path for path, count in seen.items() if count > 1]


def sanity_check_stage2(
    combined: dict[str, dict[str, list[Path]]],
    *,
    all_images: dict[str, list[Path]],
    collisions: dict[str, int],
    report_path: Path,
) -> None:
    duplicate_sources = report_duplicate_sources(all_images)
    summary = {
        "class_counts": {cls: len(paths) for cls, paths in all_images.items()},
        "splits": {
            split: {cls: len(paths) for cls, paths in class_map.items()}
            for split, class_map in combined.items()
        },
        "duplicate_sources": duplicate_sources,
        "copy_collisions": collisions,
        "dimension_summary": {
            cls: summarize_dimensions(paths)
            for cls, paths in all_images.items()
        },
    }

    missing = [
        f"{split}/{cls}"
        for split, class_map in combined.items()
        for cls, paths in class_map.items()
        if not paths
    ]
    if missing:
        raise SystemExit(f"Empty Stage 2 split detected: {', '.join(missing)}")

    if duplicate_sources:
        print(f"[warn] duplicate source paths detected: {len(duplicate_sources)}")
    if collisions:
        print(f"[warn] filename collisions handled during copy: {collisions}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[Stage 2] Sanity checks")
    print(f"  class counts : {summary['class_counts']}")
    print(f"  collisions   : {collisions or 'none'}")
    print(f"  report       : {report_path}")
    for cls, dims in summary["dimension_summary"].items():
        print(f"  dims {cls:8s}: {dims}")


def prepare_stage1(pklot_dir: Path, out_dir: Path, yaml_path: Path) -> None:
    print(f"\n[Stage 1] Building detection dataset -> {out_dir}/")
    splits = _roboflow_splits(pklot_dir)
    if not splits:
        raise SystemExit(
            f"No Roboflow splits found in {pklot_dir}. Expected train/valid/test image+label dirs."
        )

    total_images = 0
    total_boxes = 0
    for split_name, image_dir, label_dir in splits:
        out_image = out_dir / split_name / "images"
        out_label = out_dir / split_name / "labels"
        out_image.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)

        split_images = 0
        split_boxes = 0
        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            shutil.copy2(image_path, out_image / image_path.name)
            remapped = []
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                remapped.append("0 " + " ".join(parts[1:]))
                split_boxes += 1
            if remapped:
                (out_label / f"{image_path.stem}.txt").write_text("\n".join(remapped) + "\n")
                split_images += 1
        print(f"  {split_name:5s}: {split_images:5d} images  {split_boxes:7d} boxes")
        total_images += split_images
        total_boxes += split_boxes

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(out_dir.resolve()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": 1,
                "names": ["space"],
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"  total       : {total_images} images  {total_boxes} boxes")
    print(f"  yaml        : {yaml_path}")


def prepare_single_model_detection(pklot_dir: Path, out_dir: Path, yaml_path: Path) -> None:
    print(f"\n[Single Model] Building occupancy detection dataset -> {out_dir}/")
    splits = _roboflow_splits(pklot_dir)
    if not splits:
        raise SystemExit(
            f"No Roboflow splits found in {pklot_dir}. Expected train/valid/test image+label dirs."
        )

    total_images = 0
    total_boxes = 0
    for split_name, image_dir, label_dir in splits:
        out_image = out_dir / split_name / "images"
        out_label = out_dir / split_name / "labels"
        out_image.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)

        split_images = 0
        split_boxes = 0
        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            valid_lines = []
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id not in (_ROBOFLOW_FREE_IDS | _ROBOFLOW_OCC_IDS):
                    continue
                valid_lines.append(line.strip())
                split_boxes += 1
            if not valid_lines:
                continue
            shutil.copy2(image_path, out_image / image_path.name)
            (out_label / f"{image_path.stem}.txt").write_text("\n".join(valid_lines) + "\n")
            split_images += 1
        print(f"  {split_name:5s}: {split_images:5d} images  {split_boxes:7d} boxes")
        total_images += split_images
        total_boxes += split_boxes

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(out_dir.resolve()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": 2,
                "names": ["free", "occupied"],
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"  total       : {total_images} images  {total_boxes} boxes")
    print(f"  yaml        : {yaml_path}")


def collect_roboflow_patches(root: Path, patch_output: Path) -> dict[str, list[Path]]:
    patches: dict[str, list[Path]] = {"free": [], "occupied": []}
    print(f"\n[Stage 2] Cropping PKLot patches -> {patch_output}/")
    for split_name, image_dir, label_dir in _roboflow_splits(root):
        split_counts = {"free": 0, "occupied": 0}
        for class_name in ("free", "occupied"):
            (patch_output / split_name / class_name).mkdir(parents=True, exist_ok=True)
        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            with Image.open(image_path) as img:
                width, height = img.size
                for index, line in enumerate(label_path.read_text().splitlines()):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx, cy, bw, bh = (float(value) for value in parts[1:5])
                    x1 = max(0, int((cx - bw / 2) * width))
                    y1 = max(0, int((cy - bh / 2) * height))
                    x2 = min(width, int((cx + bw / 2) * width))
                    y2 = min(height, int((cy + bh / 2) * height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if class_id in _ROBOFLOW_FREE_IDS:
                        class_name = "free"
                    elif class_id in _ROBOFLOW_OCC_IDS:
                        class_name = "occupied"
                    else:
                        continue
                    target = patch_output / split_name / class_name / f"{image_path.stem}__{index:04d}.jpg"
                    img.crop((x1, y1, x2, y2)).save(target)
                    patches[class_name].append(target)
                    split_counts[class_name] += 1
        print(
            f"  {split_name:5s}: {split_counts['free']:7d} free  "
            f"{split_counts['occupied']:7d} occupied"
        )
    return patches


def collect_legacy_patches(root: Path) -> dict[str, list[Path]]:
    patches: dict[str, list[Path]] = {"free": [], "occupied": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parent = path.parent.name.lower()
        if parent in _FREE_DIRS:
            patches["free"].append(path)
        elif parent in _OCC_DIRS:
            patches["occupied"].append(path)
    return patches


def prepare_stage2(args: argparse.Namespace) -> None:
    validate_split_ratios(args.val_ratio, args.test_ratio)
    pklot_dir = Path(args.pklot_dir)
    patch_cache = Path(args.patch_cache) if args.patch_cache else pklot_dir.parent / f"{pklot_dir.name}_patches"
    stage2_output = Path(args.stage2_output)

    pklot_images = collect_roboflow_patches(pklot_dir, patch_cache)
    if not any(pklot_images.values()):
        raise SystemExit("No PKLot patches were created. Check the Roboflow export layout and labels.")

    pklot_splits = stratified_split(
        pklot_images,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    cnrpark_splits = {"train": {"free": [], "occupied": []}, "val": {"free": [], "occupied": []}, "test": {"free": [], "occupied": []}}
    cnr_source_root: Path | None = None
    cnr_images = {"free": [], "occupied": []}
    if args.cnrpark_dir:
        cnr_source_root = Path(args.cnrpark_dir)
        if not cnr_source_root.exists():
            raise SystemExit(f"CNRPark directory not found: {cnr_source_root}")
        cnr_images = collect_legacy_patches(cnr_source_root)
        print(f"\n[Stage 2] Collected CNRPark patches: free={len(cnr_images['free'])} occupied={len(cnr_images['occupied'])}")
        if any(cnr_images.values()):
            cnrpark_splits = stratified_split(
                cnr_images,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )

    combined: dict[str, dict[str, list[Path]]] = {}
    for split in ("train", "val", "test"):
        combined[split] = {}
        for class_name in ("free", "occupied"):
            combined[split][class_name] = (
                pklot_splits[split].get(class_name, []) +
                cnrpark_splits[split].get(class_name, [])
            )

    print(f"\n[Stage 2] Writing combined dataset -> {stage2_output}/")
    collisions = copy_images(combined, stage2_output, source_root=patch_cache)
    pklot_collisions = copy_test_flat(
        pklot_splits["test"],
        Path(args.pklot_test_output),
        source_root=patch_cache,
    )
    if args.cnrpark_dir and any(cnr_images.values()):
        cnr_collisions = copy_test_flat(
            cnrpark_splits["test"],
            Path(args.cnrpark_test_output),
            source_root=cnr_source_root,
        )
    else:
        cnr_collisions = {}

    all_images = {
        "free": pklot_images["free"] + cnr_images["free"],
        "occupied": pklot_images["occupied"] + cnr_images["occupied"],
    }
    sanity_check_stage2(
        combined,
        all_images=all_images,
        collisions={
            **collisions,
            **{f"pklot_test/{k}": v for k, v in pklot_collisions.items()},
            **{f"cnrpark_test/{k}": v for k, v in cnr_collisions.items()},
        },
        report_path=stage2_output / SANITY_REPORT,
    )

    print("\n[Stage 2] Split summary")
    for split_name in ("train", "val", "test"):
        counts = {cls: len(paths) for cls, paths in combined[split_name].items()}
        print(f"  {split_name:5s}: {counts}")
    pklot_counts = {cls: len(paths) for cls, paths in pklot_splits["test"].items()}
    print(f"  pklot_test : {pklot_counts}")
    if args.cnrpark_dir and any(cnr_images.values()):
        cnr_counts = {cls: len(paths) for cls, paths in cnrpark_splits["test"].items()}
        print(f"  cnrpark_test: {cnr_counts}")


def weather_split_paths(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("sunny", "cloudy", "rainy")}
    if not all(path.exists() for path in paths.values()):
        raise SystemExit(f"Per-weather evaluation requested but weather splits are unavailable. {WEATHER_CONVENTION}")
    return paths


def main() -> None:
    args = parse_args()
    if not args.stage1 and not args.stage2 and not args.single_model:
        args.stage2 = True

    pklot_dir = Path(args.pklot_dir)
    if not pklot_dir.exists():
        raise SystemExit(f"PKLot directory not found: {pklot_dir}")
    if not _roboflow_splits(pklot_dir):
        raise SystemExit(f"No Roboflow train/valid/test splits found in {pklot_dir}")

    if args.stage1:
        prepare_stage1(pklot_dir, Path(args.stage1_output), Path(args.stage1_yaml))
    if args.single_model:
        prepare_single_model_detection(
            pklot_dir,
            Path(args.single_model_output),
            Path(args.single_model_yaml),
        )
    if args.stage2:
        prepare_stage2(args)


if __name__ == "__main__":
    main()
