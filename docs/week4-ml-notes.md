# Week 4 ML Notes

Week 4 should frame Stage 2 classification as the main ML track, not full-scene PKLot detection.

The repo also supports a single-model full-frame occupancy detector as an ML-only baseline, but it is not the deployed default path.

## Dataset Story

- primary source: PKLot Roboflow export with spot boxes used to crop parking patches
- optional source: CNRPark-EXT patch folders merged into Stage 2 training
- canonical outputs: `stage2_data/`, `pklot_test/`, `cnrpark_test/`

## Stage 2 Training Path

```bash
python ml/prepare_dataset.py --stage2 --pklot-dir /path/to/pklot_roboflow
python ml/train.py --stage2 --variant n --device mps
python ml/train.py --stage2 --variant s --device mps
python ml/train.py --stage2 --variant m --device mps
```

## Other Supported ML Tracks

Separate Stage 1 detector:

```bash
python ml/prepare_dataset.py --stage1 --pklot-dir /path/to/pklot_roboflow
python ml/train.py --stage1 --variant n --device mps
```

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
