# Final Runbook

This runbook matches the final required architecture:

`trained Stage 1 detector -> crop -> trained Stage 2 classifier -> smoothing -> JSON -> FastAPI`

## 1. Prepare Datasets

```bash
source .venv/bin/activate
python ml/prepare_dataset.py --stage1 --pklot-dir datasets/pklot_raw
python ml/prepare_dataset.py --stage2 --pklot-dir datasets/pklot_raw
```

Add `--cnrpark-dir <path>` to the Stage 2 command when the CNRPark-EXT patch folders are available.
The Stage 2 prep command accepts either the official `cnrpark.it` `PATCHES/` + `LABELS/` archive layout or pre-flattened `free/` / `occupied/` folders. When weather labels are present it also writes `datasets/stage2_weather/` for per-weather evaluation.

## 2. Train Models

Stage 1 detector:

```bash
python ml/train.py --stage1 --variant s --device mps
```

Stage 2 classifier comparison:

```bash
python ml/train.py --stage2 --variant n --device mps
python ml/train.py --stage2 --variant s --device mps
python ml/train.py --stage2 --variant m --device mps
```

## 3. Evaluate

```bash
python ml/evaluate.py --stage1 --weights runs/stage1_det/yolov8s_stage1/weights/best.pt --split val --device mps
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8s_stage2/weights/best.pt --split val --device mps
python ml/evaluate.py --stage2 --split val --device mps --compare \
  runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  runs/stage2_cls/yolov8s_stage2/weights/best.pt \
  runs/stage2_cls/yolov8m_stage2/weights/best.pt
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8s_stage2/weights/best.pt --split val --device mps --sweep
```

Cross-dataset evaluation, when exports are available:

```bash
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8s_stage2/weights/best.pt --cross-dataset pklot_test --device mps
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8s_stage2/weights/best.pt --cross-dataset cnrpark_test --device mps
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8s_stage2/weights/best.pt --data datasets/stage2_weather --per-weather --device mps
```

## 4. Export + Benchmark

```bash
python ml/export.py --weights runs/stage2_cls/yolov8s_stage2/weights/best.pt --imgsz 64
python edge/benchmark.py \
  --task classify \
  --image samples/demo.jpg \
  --model runs/stage2_cls/yolov8s_stage2/weights/best.pt \
  --imgsz 64 \
  --roi 50 100 200 250
python ml/bandwidth.py
```

## 5. End-to-End Validation

Start backend:

```bash
uvicorn backend.main:app --reload
```

Run integrated demo:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage1-detector \
  --stage1-model runs/stage1_det/yolov8s_stage1/weights/best.pt \
  --stage2-model runs/stage2_cls/yolov8s_stage2/weights/best.pt \
  --post \
  --save-annotated logs/final-demo-annotated.jpg
```

Run stability test:

```bash
python edge/stability_test.py \
  --image samples/demo.jpg \
  --stage1-detector \
  --stage1-model runs/stage1_det/yolov8s_stage1/weights/best.pt \
  --stage2-model runs/stage2_cls/yolov8s_stage2/weights/best.pt \
  --post \
  --duration 1800
```

## 6. Final Packaging

```bash
python ml/finalize.py
```

This generates:

- `artifacts/final_manifest.json`
- `docs/final-artifact-summary.md`
