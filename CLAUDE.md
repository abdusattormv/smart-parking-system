# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Camera-based smart parking prototype comparing YOLOv8n/s/m on the PKLot dataset, evaluating INT8 quantization for edge deployment, with a minimal FastAPI backend for logging. There is no frontend — the focus is ML quality and edge inference reliability.

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

**Run static-image inference demo:**
```bash
python edge/detect.py --image /path/to/parking.jpg
python edge/detect.py --image /path/to/parking.jpg --post              # also POST to backend
python edge/detect.py --image /path/to/parking.jpg --save-annotated logs/out.jpg
python edge/detect.py --image /path/to/parking.jpg --device cpu        # if MPS unavailable
python edge/detect.py --image /path/to/parking.jpg --pklot-model       # use PKLot-trained model
```

**ML pipeline:**
```bash
python ml/prepare_dataset.py --pklot-dir datasets/pklot_raw   # convert + split PKLot
python ml/train.py --variant n                                 # train YOLOv8n
python ml/export.py --weights runs/parking/yolov8n_pklot/weights/best.pt
python ml/evaluate.py --weights artifacts/models/best.pt --full
python ml/evaluate.py --weights artifacts/models/best.pt --per-weather
python ml/evaluate.py --compare runs/parking/*/weights/best.pt
python ml/bandwidth.py
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

**`edge/detect.py`** — inference pipeline. Loads a YOLO model (default `yolov8n.pt`), runs inference on a static image, maps vehicle detections to mock parking spot occupancy (`spot_1..spot_n`), and optionally POSTs a JSON payload to the backend. Car class IDs are `{2, 3, 5, 7}` (COCO). Device defaults to `mps` with CPU as fallback.

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
