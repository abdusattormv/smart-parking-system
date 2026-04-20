# Week 4 ML Notes

Week 4 should frame Stage 2 classification as the main ML track, not full-scene PKLot detection.

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
python ml/evaluate.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --split val
python ml/evaluate.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --cross-dataset pklot_test
```

Per-weather evaluation is only valid when the dataset is arranged as:

```text
<dataset-root>/sunny/<class>/*
<dataset-root>/cloudy/<class>/*
<dataset-root>/rainy/<class>/*
```

If that layout is missing, the repo should fail clearly instead of inventing weather labels.
