# Week 4 ML Notes

Week 4 should frame the two-stage full-frame pipeline as the main ML track:

`full-frame slot detector -> slot crop -> occupancy classifier`

The repo also supports a single-model full-frame occupancy detector as an ML-only baseline, but it is not the deployed default path.

PKLot full-frame detection must be evaluated with scene holdout. Random image splits are not accepted as generalization evidence here.

PKLot full-frame detection should be treated carefully because Roboflow exports may include duplicate `.rf.*` variants and zero-label frames. Detection prep now deduplicates by normalized frame id, rebuilds train/val/test by scene group, excludes zero-label frames, and writes leakage plus annotation-audit details to the dataset report.

## Dataset Story

- primary source: PKLot Roboflow export with full-frame slot labels
- optional source: CNRPark-EXT patch folders merged into Stage 2 training
- canonical outputs: `stage2_data/`, `pklot_test/`, `cnrpark_test/`

## Recommended Stage 1 Training Path

```bash
python ml/prepare_dataset.py --stage1 --pklot-dir /path/to/pklot_roboflow
python ml/train.py --stage1 --variant s --device mps
python ml/train.py --stage1 --variant m --imgsz 960 --device mps
python ml/evaluate.py --stage1 --weights runs/stage1_det/yolov8s_stage1/weights/best.pt --split val
```

## Stage 2 Training Path

```bash
python ml/prepare_dataset.py --stage2 --pklot-dir /path/to/pklot_roboflow
python ml/train.py --stage2 --variant n --device mps
python ml/train.py --stage2 --variant s --device mps
python ml/train.py --stage2 --variant m --device mps
```

## Other Supported ML Tracks

Single-model occupancy detector baseline:

```bash
python ml/prepare_dataset.py --single-model --pklot-dir /path/to/pklot_roboflow
python ml/train.py --single-model --variant n --device mps
python ml/evaluate.py --single-model --weights runs/single_model_det/yolov8n_single_model/weights/best.pt --split val
```

## Comparison Matrix

The main comparison is `yolov8n-cls` vs `yolov8s-cls` vs `yolov8m-cls` on:

- top-1 accuracy
- precision
- recall
- F1
- checkpoint size
- deployment latency / FPS

## Evaluation Path

```bash
python ml/evaluate.py --stage1 --weights runs/stage1_det/yolov8s_stage1/weights/best.pt --split val
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --split val
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --cross-dataset pklot_test
```

Per-weather evaluation is only valid when the dataset is arranged as:

```text
<dataset-root>/sunny/<class>/*
<dataset-root>/cloudy/<class>/*
<dataset-root>/rainy/<class>/*
```

If that layout is missing, the repo should fail clearly instead of inventing weather labels.
