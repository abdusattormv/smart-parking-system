#!/usr/bin/env python3
"""Prepare the PKLot dataset for YOLOv8 training.

Steps:
  1. Walk the PKLot directory tree (lot / weather / date / images+xmls).
  2. Parse each XML and convert rotated-rect annotations to YOLO axis-aligned
     bounding boxes (class 0 = empty, class 1 = occupied).
  3. Apply a stratified 70/15/15 split across (lot, weather) groups.
  4. Write images and labels into datasets/pklot/{images,labels}/{train,val,test}.
  5. Create per-weather symlink splits under datasets/pklot/test_{sunny,cloudy,rainy}.
  6. Write ml/data.yaml.

Usage:
  python ml/prepare_dataset.py --pklot-dir /path/to/PKLot --output-dir datasets/pklot
  python ml/prepare_dataset.py --help
"""

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

# PKLot class mapping
CLASS_EMPTY = 0
CLASS_OCCUPIED = 1

WEATHER_MAP = {
    "Sunny": "sunny",
    "Cloudy": "cloudy",
    "Rainy": "rainy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare PKLot for YOLOv8 training.")
    p.add_argument(
        "--pklot-dir",
        default="datasets/pklot_raw",
        help="Root of the downloaded PKLot directory (contains PUCPR/, UFPR04/, UFPR05/).",
    )
    p.add_argument(
        "--output-dir",
        default="datasets/pklot",
        help="Output root for the prepared dataset.",
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation split ratio."
    )
    p.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test split ratio."
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    p.add_argument(
        "--data-yaml",
        default="ml/data.yaml",
        help="Output path for the Ultralytics data.yaml.",
    )
    return p.parse_args()


def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse one PKLot XML and return YOLO annotation lines.

    PKLot rotatedRect stores center + size + angle. We convert to an
    axis-aligned bounding box (worst-case extent) since YOLOv8 uses xyxy/xywh.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for space in root.findall(".//space"):
        occupied = int(space.get("occupied", 0))
        cls = CLASS_OCCUPIED if occupied else CLASS_EMPTY

        rect = space.find("rotatedRect")
        if rect is None:
            # Fallback: use contour bounding box
            pts = [
                (float(pt.get("x")), float(pt.get("y")))
                for pt in space.findall(".//point")
            ]
            if not pts:
                continue
            xs, ys = zip(*pts)
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
        else:
            center = rect.find("center")
            size = rect.find("size")
            angle_el = rect.find("angle")
            cx = float(center.get("x"))
            cy = float(center.get("y"))
            rw = float(size.get("w"))
            rh = float(size.get("h"))
            angle_deg = float(angle_el.get("d")) if angle_el is not None else 0.0
            # Axis-aligned bounding box of the rotated rect
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
            w = rw * cos_a + rh * sin_a
            h = rw * sin_a + rh * cos_a

        # Normalize to [0, 1]
        nx = cx / img_w
        ny = cy / img_h
        nw = w / img_w
        nh = h / img_h

        # Clamp to valid range
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        if nw > 0 and nh > 0:
            lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    return lines


def collect_samples(pklot_dir: Path) -> list[dict]:
    """Walk the PKLot tree and collect (image_path, xml_path, lot, weather) records."""
    samples = []
    for lot_dir in sorted(pklot_dir.iterdir()):
        if not lot_dir.is_dir():
            continue
        lot = lot_dir.name
        for weather_dir in sorted(lot_dir.iterdir()):
            if not weather_dir.is_dir():
                continue
            weather_raw = weather_dir.name
            weather = WEATHER_MAP.get(weather_raw, weather_raw.lower())
            for date_dir in sorted(weather_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                for img_path in sorted(date_dir.glob("*.jpg")):
                    xml_path = img_path.with_suffix(".xml")
                    if xml_path.exists():
                        samples.append(
                            {
                                "image": img_path,
                                "xml": xml_path,
                                "lot": lot,
                                "weather": weather,
                                "group": f"{lot}_{weather}",
                            }
                        )
    return samples


def copy_sample(
    sample: dict,
    split: str,
    output_dir: Path,
) -> None:
    """Copy image and write YOLO label for one sample into the split directory."""
    import cv2

    img_path: Path = sample["image"]
    xml_path: Path = sample["xml"]
    stem = img_path.stem

    img_out = output_dir / "images" / split / img_path.name
    lbl_out = output_dir / "labels" / split / f"{stem}.txt"

    img_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, img_out)

    frame = cv2.imread(str(img_path))
    if frame is None:
        h, w = 720, 1280  # fallback
    else:
        h, w = frame.shape[:2]

    lines = xml_to_yolo(xml_path, w, h)
    lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""))


def make_weather_test_splits(output_dir: Path, test_samples: list[dict]) -> None:
    """Create per-weather symlink directories under output_dir/test_{weather}."""
    weathers = {s["weather"] for s in test_samples}
    for weather in sorted(weathers):
        w_img = output_dir / f"test_{weather}" / "images"
        w_lbl = output_dir / f"test_{weather}" / "labels"
        w_img.mkdir(parents=True, exist_ok=True)
        w_lbl.mkdir(parents=True, exist_ok=True)

        for s in test_samples:
            if s["weather"] != weather:
                continue
            stem = s["image"].stem
            src_img = output_dir / "images" / "test" / s["image"].name
            src_lbl = output_dir / "labels" / "test" / f"{stem}.txt"
            dst_img = w_img / s["image"].name
            dst_lbl = w_lbl / f"{stem}.txt"
            if src_img.exists() and not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            if src_lbl.exists() and not dst_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)


def write_data_yaml(output_dir: Path, yaml_path: Path) -> None:
    abs_out = output_dir.resolve()
    data = {
        "path": str(abs_out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": ["empty", "occupied"],
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml written to {yaml_path}")


def main() -> None:
    args = parse_args()
    pklot_dir = Path(args.pklot_dir)
    output_dir = Path(args.output_dir)

    if not pklot_dir.exists():
        raise SystemExit(
            f"PKLot directory not found: {pklot_dir}\n"
            "Download from Kaggle: ammarnassanalhajali/pklot-dataset\n"
            "  pip install kaggle\n"
            "  kaggle datasets download -d ammarnassanalhajali/pklot-dataset\n"
            "  unzip pklot-dataset.zip -d datasets/pklot_raw"
        )

    print(f"Scanning {pklot_dir} ...")
    samples = collect_samples(pklot_dir)
    if not samples:
        raise SystemExit(f"No samples found in {pklot_dir}. Check directory structure.")

    print(f"Found {len(samples)} samples across lots/weathers.")

    # Stratified split by (lot, weather) group
    groups = [s["group"] for s in samples]
    train_val, test = train_test_split(
        samples,
        test_size=args.test_ratio,
        stratify=groups,
        random_state=args.seed,
    )
    val_ratio_adjusted = args.val_ratio / (1.0 - args.test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=[s["group"] for s in train_val],
        random_state=args.seed,
    )

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    for split_name, split_samples in [("train", train), ("val", val), ("test", test)]:
        print(f"Copying {split_name} ({len(split_samples)} images) ...")
        for i, sample in enumerate(split_samples):
            copy_sample(sample, split_name, output_dir)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(split_samples)}")

    print("Creating per-weather test splits ...")
    make_weather_test_splits(output_dir, test)

    write_data_yaml(output_dir, Path(args.data_yaml))

    print("\nDone.")
    print(f"  Train: {len(train)} images → {output_dir}/images/train/")
    print(f"  Val:   {len(val)} images → {output_dir}/images/val/")
    print(f"  Test:  {len(test)} images → {output_dir}/images/test/")


if __name__ == "__main__":
    main()
