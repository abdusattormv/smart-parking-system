# Smart Parking System

This repo implements the v3 smart parking pipeline:

`static camera -> fixed ROIs -> per-spot crop -> YOLOv8*-cls -> temporal smoothing -> JSON -> FastAPI`

The deployed demo still supports fixed ROIs for a static camera, but the recommended ML workflow is now:

`full-frame slot detector -> per-slot crop -> occupancy classifier`

Stage 1 full-frame slot detection is the primary generalization track. Stage 2 patch classification remains the occupancy model. The single-model full-frame occupancy detector remains only as an ML comparison baseline.

The final-project default is now:

`trained Stage 1 detector -> per-slot crop -> trained Stage 2 classifier -> smoothing -> JSON -> FastAPI`

## Repo Focus

- `edge/` runs the two-stage edge pipeline and emits the v3 payload contract
- `ml/` prepares Stage 2 datasets, trains YOLOv8 classification models, and evaluates classification metrics
- `backend/` stores payloads from the edge pipeline and returns latest/history views
- `docs/` contains the canonical PRD and aligned milestone notes

## Canonical Artifacts

- Stage 2 dataset: `stage2_data/`
- Weather export for CNR evaluation: `datasets/stage2_weather/`
- Cross-dataset exports: `datasets/pklot_test/`, `datasets/cnrpark_test/`
- Stage 2 training handoff: `runs/stage2_cls/.../weights/best.pt`
- Backend payload:

```json
{
  "spots": {
    "spot_1": "free",
    "spot_2": "occupied"
  },
  "confidence": {
    "spot_1": 0.91,
    "spot_2": 0.84
  },
  "timestamp": "2026-04-21T00:00:00Z"
}
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## ML Workflows

Full-frame detection must be evaluated by scene holdout. Image-level random splits are not treated as evidence of generalization in this repo.

Recommended Stage 1 full-frame slot detector:

```bash
python ml/prepare_dataset.py --stage1 --pklot-dir /path/to/pklot_roboflow
python ml/train.py --stage1 --variant s --device mps
python ml/train.py --stage1 --variant m --imgsz 960 --device mps
python ml/evaluate.py --stage1 --weights runs/stage1_det/yolov8s_stage1/weights/best.pt --split val
```

Single-model occupancy detector baseline:

```bash
python ml/prepare_dataset.py --single-model \
  --pklot-dir datasets/pklot_v4 \
  --single-model-output single_model_data_boxes \
  --single-model-yaml ml/single_model_boxes.yaml

python ml/train.py --single-model --variant n --device mps
python ml/evaluate.py --single-model --weights runs/single_model_det/yolov8n_single_model/weights/best.pt --split val
```

Stage 2 occupancy classifier with Stage 1-derived crops:

Prepare the classification dataset:

```bash
python ml/prepare_dataset.py --stage2 --pklot-dir /path/to/pklot_roboflow
python ml/prepare_dataset.py --stage2 --pklot-dir /path/to/pklot_roboflow --cnrpark-dir /path/to/cnrpark_ext
```

`--cnrpark-dir` now supports both:

- pre-flattened patch folders with `free/` and `occupied/` subdirectories
- the official `cnrpark.it` archive layout with `PATCHES/` and `LABELS/`

When weather labels are available, dataset prep also exports `datasets/stage2_weather/{sunny,cloudy,rainy}/{free,occupied}` for per-weather evaluation.

Train the main classifier comparison set:

```bash
python ml/train.py --stage2 --variant n --device mps
python ml/train.py --stage2 --variant s --device mps
python ml/train.py --stage2 --variant m --device mps
```

Current final artifact choice:

- deployed Stage 1 detector path: `runs/stage1_det/yolov8s_stage1/weights/best.pt`
- strongest Stage 2 classifier from the saved comparison logs: `runs/stage2_cls/yolov8m_stage2/weights/best.pt`
- current exported inference artifacts: `artifacts/models/best.pt`, `artifacts/models/best.onnx`, `artifacts/models/best_int8.onnx`

Generate a final artifact summary after training and evaluation:

```bash
python ml/finalize.py
```

Accuracy notes:

- the saved `runs/stage2_cls/yolov8n_stage2/results.csv` shows an early accuracy collapse after epoch 4, so the repo now uses a lower Stage 2 learning rate, longer patience, cosine LR decay, and classifier dropout by default
- Stage 1 and single-model prep now rebuild train/val/test by scene holdout after deduplicating `.rf.*` Roboflow variants
- PKLot full-frame prep excludes zero-label frames and writes `detection_dataset_report.json` with split leakage checks and annotation-audit summaries
- Roboflow PKLot exports may use per-slot polygons; `ml/prepare_dataset.py` converts those polygons into clipped YOLO detection boxes automatically for Stage 1, single-model detection, and Stage 2 patch cropping

Evaluate Stage 2 classification:

```bash
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8m_stage2/weights/best.pt --split val --device mps --batch 256
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8m_stage2/weights/best.pt --cross-dataset datasets/pklot_test --device mps --batch 256
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8m_stage2/weights/best.pt --cross-dataset datasets/cnrpark_test --device mps --batch 256
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8m_stage2/weights/best.pt --data datasets/stage2_weather --per-weather --device mps --batch 512
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8m_stage2/weights/best.pt --split val --device mps --batch 256 --sweep
python ml/evaluate.py --stage2 --compare \
  runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  runs/stage2_cls/yolov8s_stage2/weights/best.pt \
  runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --split val --device mps --batch 256
```

The saved threshold sweep currently selects `0.1` as the best offline validation threshold for `yolov8m_stage2`.
The deployed edge config still uses `0.3`, which matches the saved comparison run and is less aggressive for live demos.

Run one-off prediction:

```bash
python ml/predict.py --weights runs/stage2_cls/yolov8m_stage2/weights/best.pt --source samples/demo.jpg
python ml/predict.py --stage1 --weights runs/stage1_det/yolov8s_stage1/weights/best.pt --source samples/demo.jpg
python ml/predict.py --single-model --weights runs/detect/train/weights/best.pt --source samples/demo.jpg
```

The deployed/default runtime remains two-stage. Fixed ROIs are still available for the static-camera demo, but they are deployment-specific rather than the repo’s generalizable ML recommendation.

## Edge Demo

Start the backend:

```bash
uvicorn backend.main:app --reload
```

Open the live MJPEG stream in a browser:

```html
<img src="http://127.0.0.1:8000/stream" alt="Parking stream">
```

Run image inference with fixed ROIs:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --save-annotated logs/demo-annotated.jpg
```

`detect.py` now posts to the backend by default, so `/status` and `/history` update automatically while the backend is running.
Use `--no-post` for offline-only inference.

Run the integrated final pipeline with the trained Stage 1 detector:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage1-detector \
  --stage1-model runs/stage1_det/yolov8s_stage1/weights/best.pt \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --save-annotated logs/final-demo-annotated.jpg
```

Benchmark the Stage 2 classifier on a representative ROI patch:

```bash
python edge/benchmark.py \
  --task classify \
  --image samples/demo.jpg \
  --model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --imgsz 64 \
  --roi 50 100 200 250
```

Run the short reproducible stability check that generated the current summary:

```bash
python edge/stability_test.py \
  --image samples/demo.jpg \
  --stage1-detector \
  --stage1-model runs/stage1_det/yolov8s_stage1/weights/best.pt \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --device mps \
  --duration 15 \
  --frame-interval 500
```

For the longer Week 7 soak test, keep the same command and raise `--duration` to `1800`.

Run live camera inference:

```bash
python edge/detect.py --camera 0 --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt
python edge/detect.py --camera iphone --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt
```

`--camera iphone` is macOS-only and targets Continuity Camera / attached iPhone cameras.
Camera mode updates `logs/latest_frame.jpg` continuously so `/stream` can render the latest annotated frame without waiting for POST intervals.

## Docs

- Canonical PRD: [docs/prd.md](/Users/thebkht/Projects/smart-parking-system/docs/prd.md)
- Docs index: [docs/README.md](/Users/thebkht/Projects/smart-parking-system/docs/README.md)
- Edge details: [edge/README.md](/Users/thebkht/Projects/smart-parking-system/edge/README.md)
- Backend contract: [backend/README.md](/Users/thebkht/Projects/smart-parking-system/backend/README.md)
