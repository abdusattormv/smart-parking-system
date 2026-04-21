# Smart Parking System

This repo implements the v3 smart parking pipeline:

`static camera -> fixed ROIs -> per-spot crop -> YOLOv8*-cls -> temporal smoothing -> JSON -> FastAPI`

Fixed ROIs are the default Stage 1 path because the target demo camera is static. An optional YOLO Stage 1 detector remains available for experiments, Stage 2 patch classification is the main ML accuracy track, and a single-model full-frame occupancy detector is available as an ML-only comparison baseline.

## Repo Focus

- `edge/` runs the two-stage edge pipeline and emits the v3 payload contract
- `ml/` prepares Stage 2 datasets, trains YOLOv8 classification models, and evaluates classification metrics
- `backend/` stores payloads from the edge pipeline and returns latest/history views
- `docs/` contains the canonical PRD and aligned milestone notes

## Canonical Artifacts

- Stage 2 dataset: `stage2_data/`
- Cross-dataset exports: `pklot_test/`, `cnrpark_test/`
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

Separate Stage 1 detector dataset:

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

Primary Stage 2 classifier workflow:

Prepare the classification dataset:

```bash
python ml/prepare_dataset.py --stage2 --pklot-dir /path/to/pklot_roboflow
python ml/prepare_dataset.py --stage2 --pklot-dir /path/to/pklot_roboflow --cnrpark-dir /path/to/cnrpark_ext
```

Train the main classifier comparison set:

```bash
python ml/train.py --stage2 --variant n --device mps
python ml/train.py --stage2 --variant s --device mps
python ml/train.py --stage2 --variant m --device mps
```

Accuracy notes:

- the saved `runs/stage2_cls/yolov8n_stage2/results.csv` shows an early accuracy collapse after epoch 4, so the repo now uses a lower Stage 2 learning rate, longer patience, cosine LR decay, and classifier dropout by default
- PKLot full-frame exports include zero-label frames, so detection-style prep now excludes those frames and writes a detection dataset report under the output directory

Evaluate Stage 2 classification:

```bash
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --split val
python ml/evaluate.py --stage2 --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --cross-dataset pklot_test
python ml/evaluate.py --stage2 --compare \
  runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  runs/stage2_cls/yolov8s_stage2/weights/best.pt \
  runs/stage2_cls/yolov8m_stage2/weights/best.pt
```

Run one-off prediction:

```bash
python ml/predict.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --source samples/demo.jpg
python ml/predict.py --stage1 --weights runs/stage1_det/yolov8n_stage1/weights/best.pt --source samples/demo.jpg
python ml/predict.py --single-model --weights runs/detect/train/weights/best.pt --source samples/demo.jpg
```

The deployed/default runtime remains two-stage. The single-model detector exists only for ML-side comparison against the separate Stage 1 + Stage 2 approach.

## Edge Demo

Start the backend:

```bash
uvicorn backend.main:app --reload
```

Run image inference with fixed ROIs:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  --post \
  --save-annotated logs/demo-annotated.jpg
```

Run live camera inference:

```bash
python edge/detect.py --camera 0 --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt --post
python edge/detect.py --camera iphone --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt --post
```

`--camera iphone` is macOS-only and targets Continuity Camera / attached iPhone cameras.

## Docs

- Canonical PRD: [docs/prd.md](/Users/thebkht/Projects/smart-parking-system/docs/prd.md)
- Docs index: [docs/README.md](/Users/thebkht/Projects/smart-parking-system/docs/README.md)
- Edge details: [edge/README.md](/Users/thebkht/Projects/smart-parking-system/edge/README.md)
- Backend contract: [backend/README.md](/Users/thebkht/Projects/smart-parking-system/backend/README.md)
