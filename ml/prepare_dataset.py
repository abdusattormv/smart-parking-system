#!/usr/bin/env python3
"""Prepare the PKLot dataset for YOLOv8 training.

Supports two input formats:

  Roboflow COCO format (ammarnassanalhajali/pklot-dataset from Kaggle):
    pklot_raw/
      train/ _annotations.coco.json  *.jpg
      valid/ _annotations.coco.json  *.jpg
      test/  _annotations.coco.json  *.jpg

  Original PKLot XML format:
    pklot_raw/
      PUCPR/ Cloudy/ 2012-09-12/ *.jpg  *.xml
      UFPR04/ ...
      UFPR05/ ...

Steps:
  1. Detect format and parse annotations.
  2. Convert to YOLO txt (class 0=empty, class 1=occupied, normalized xywh).
  3. Write to datasets/pklot/{images,labels}/{train,val,test}.
  4. Write ml/data.yaml.

Usage:
  python ml/prepare_dataset.py --pklot-dir datasets/pklot_raw --output-dir datasets/pklot
  python ml/prepare_dataset.py --help
"""

import argparse
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

# PKLot class mapping (YOLO output)
CLASS_EMPTY = 0
CLASS_OCCUPIED = 1

WEATHER_MAP = {
    "Sunny": "sunny",
    "Cloudy": "cloudy",
    "Rainy": "rainy",
}

# Roboflow COCO category_id → YOLO class
COCO_CATEGORY_MAP = {
    1: CLASS_EMPTY,     # space-empty
    2: CLASS_OCCUPIED,  # space-occupied
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


def is_roboflow_format(pklot_dir: Path) -> bool:
    """Return True if this looks like the Roboflow COCO export (train/valid/test + json)."""
    return (pklot_dir / "train" / "_annotations.coco.json").exists()


def convert_roboflow_split(
    src_dir: Path,
    split_name: str,
    output_dir: Path,
) -> int:
    """Convert one Roboflow COCO split to YOLO format. Returns image count."""
    coco_path = src_dir / "_annotations.coco.json"
    with open(coco_path) as f:
        coco = json.load(f)

    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Build image id → metadata map
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        cls = COCO_CATEGORY_MAP.get(ann["category_id"])
        if cls is None:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append((cls, ann["bbox"]))

    count = 0
    for img_meta in coco["images"]:
        img_id = img_meta["id"]
        fname = img_meta["file_name"]
        w = img_meta["width"]
        h = img_meta["height"]

        src_img = src_dir / fname
        if not src_img.exists():
            continue

        shutil.copy2(src_img, img_dir / fname)

        # COCO bbox: [x_min, y_min, width, height] → YOLO: [cx, cy, nw, nh]
        lines = []
        for cls, bbox in ann_by_image.get(img_id, []):
            x, y, bw, bh = bbox
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            if nw > 0 and nh > 0:
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(fname).stem
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        count += 1

    return count


def collect_samples(pklot_dir: Path) -> list[dict]:
    """Walk the original PKLot XML tree and collect sample records."""
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


def copy_sample(sample: dict, split: str, output_dir: Path) -> None:
    """Copy image and write YOLO label for one XML-format sample."""
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
    h, w = frame.shape[:2] if frame is not None else (720, 1280)

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
            "  kaggle datasets download -d ammarnassanalhajali/pklot-dataset\n"
            "  mkdir -p datasets/pklot_raw && unzip pklot-dataset.zip -d datasets/pklot_raw"
        )

    if is_roboflow_format(pklot_dir):
        print(f"Detected Roboflow COCO format in {pklot_dir}")
        # Map Roboflow split names to standard names
        split_map = {"train": "train", "valid": "val", "test": "test"}
        totals = {}
        for src_name, dst_name in split_map.items():
            src_dir = pklot_dir / src_name
            if not src_dir.exists():
                print(f"  Skipping {src_name} (not found)")
                continue
            print(f"  Converting {src_name} → {dst_name} ...")
            count = convert_roboflow_split(src_dir, dst_name, output_dir)
            totals[dst_name] = count
            print(f"  {count} images")
    else:
        print(f"Detected original PKLot XML format in {pklot_dir}")
        print(f"Scanning ...")
        samples = collect_samples(pklot_dir)
        if not samples:
            raise SystemExit(
                f"No samples found in {pklot_dir}.\n"
                "Expected either:\n"
                "  Roboflow: train/_annotations.coco.json\n"
                "  Original: PUCPR/Sunny/date/*.jpg + *.xml"
            )
        print(f"Found {len(samples)} samples.")

        groups = [s["group"] for s in samples]
        train_val, test = train_test_split(
            samples, test_size=args.test_ratio, stratify=groups, random_state=args.seed
        )
        val_ratio_adjusted = args.val_ratio / (1.0 - args.test_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            stratify=[s["group"] for s in train_val],
            random_state=args.seed,
        )

        for split_name, split_samples in [("train", train), ("val", val), ("test", test)]:
            print(f"Copying {split_name} ({len(split_samples)} images) ...")
            for i, sample in enumerate(split_samples):
                copy_sample(sample, split_name, output_dir)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(split_samples)}")

        totals = {"train": len(train), "val": len(val), "test": len(test)}

    write_data_yaml(output_dir, Path(args.data_yaml))

    print("\nDone.")
    for split, n in totals.items():
        print(f"  {split:5s}: {n} images → {output_dir}/images/{split}/")


if __name__ == "__main__":
    main()
