# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-stage edge inference system for smart parking. Stage 1 locates spots via fixed ROI polygons (or a YOLO detector). Stage 2 classifies each cropped patch as occupied/free using YOLOv8-cls trained on PKLot + CNRPark-EXT. Only a JSON result (~1 KB) is sent to a minimal FastAPI backend — no raw video leaves the device.

## Environment

Single shared `.venv` at the repo root (Python 3.9, arm64 Mac):

```bash
make install-dev          # create .venv and install all deps including dev tools
source .venv/bin/activate # activate before running anything
```

Verify the environment:
```bash
python -c "import cv2, ultralytics, yaml; print('env ok')"
```

## Common Commands

**Run the backend:**
```bash
uvicorn backend.main:app --reload
```

**Run static-image inference demo (two-stage pipeline):**
```bash
python edge/detect.py --image /path/to/parking.jpg
python edge/detect.py --image /path/to/parking.jpg --post                              # POST to backend
python edge/detect.py --image /path/to/parking.jpg --save-annotated logs/out.jpg
python edge/detect.py --image /path/to/parking.jpg --device cpu                        # if MPS unavailable
python edge/detect.py --image /path/to/parking.jpg --stage2-model stage2_cls/weights/best.pt
python edge/detect.py --image /path/to/parking.jpg --no-fixed-roi                      # use Stage 1 YOLO detector
python edge/detect.py --camera 0                                                        # live camera mode
```

**ML pipeline (Stage 2 — PKLot + CNRPark-EXT classifier):**
```bash
python ml/prepare_dataset.py --pklot-dir datasets/pklot_patches                        # build stage2_data/
python ml/prepare_dataset.py --pklot-dir datasets/pklot_patches --cnrpark-dir datasets/cnrpark
python ml/train.py --stage2 --variant n                                                 # train YOLOv8n-cls
python ml/train.py --stage2 --variant s                                                 # train YOLOv8s-cls
python ml/train.py --stage2 --variant m                                                 # train YOLOv8m-cls
python ml/export.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt
python ml/evaluate.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --full
python ml/evaluate.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --cross-dataset cnrpark_test
python ml/evaluate.py --compare runs/stage2_cls/*/weights/best.pt
python ml/bandwidth.py
```

**ML pipeline (clf-data workflow — SVM + YOLO comparison):**
```bash
python ml/prepare_dataset.py --clf-dir datasets/clf-data
python ml/train.py --variant n
```

**Lint / format:**
```bash
ruff check .
black .
```

**Run tests:**
```bash
pytest
```

## Architecture

Two main components:

**`edge/detect.py`** — two-stage inference pipeline.
- Stage 1: `get_spot_boxes()` returns `(x1,y1,x2,y2)` per spot. Default: fixed `FIXED_ROIS` dict. With `--no-fixed-roi`: runs a YOLO spot detector.
- Stage 2: `classify_patch()` crops each ROI to 64×64 and runs YOLOv8-cls. Default model: `yolov8n-cls.pt` (placeholder; replace with trained `stage2_cls/weights/best.pt`).
- Output JSON: `{spot_1: "free", confidence: {spot_1: 0.97}, timestamp: "..."}`.
- Device defaults to `mps` with CPU fallback via `--device cpu`.

**`backend/main.py`** — minimal FastAPI app. Persists every update to `parking.db` (SQLite):
- `POST /update` — accepts any JSON payload (open schema via `extra="allow"`), stores to DB
- `GET /status` — returns the latest stored update
- `GET /history` — returns up to N recent updates (default 100)
- `GET /health` — liveness check

**`edge/config.example.yaml`** — template for local `edge/config.yaml` (gitignored). Covers model path, input mode (image or camera), postprocessing (smoothing window, overlap threshold), and logging output format.

**`ml/`** — ML pipeline scripts: `prepare_dataset.py`, `train.py`, `export.py`, `evaluate.py`, `bandwidth.py`.

**`artifacts/models/`** — where trained model checkpoints (`best.pt`, `best.onnx`, `best_int8.onnx`) are placed. Not committed.

**`logs/`** — timestamped CSV/JSON inference logs. Not committed.

## Conventions

- Model artifacts go in `artifacts/models/`; logs go in `logs/`
- Local edge config: copy `edge/config.example.yaml` → `edge/config.yaml`
- `requirements.txt` = runtime stack; `requirements-dev.txt` adds pytest, black, ruff, jupyter
- `opencv-python` (not `headless`) is used intentionally for webcam support
