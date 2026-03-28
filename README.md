# Smart Parking System

An intelligent edge computing project for parking occupancy detection with a stronger focus on ML quality, model comparison, and on-device reliability.

## Project Story

This project is a camera-based smart parking prototype that compares YOLOv8n, YOLOv8s, and YOLOv8m on PKLot, evaluates INT8 quantization for edge deployment, and demonstrates a stable on-device inference pipeline with minimal backend logging.

The system follows a privacy-first design:

- video stays on the edge device
- inference runs locally on a MacBook
- results are recorded and optionally logged through a minimal FastAPI backend
- a full web dashboard is out of scope in favor of ML and edge improvements

## Core Tracks

- `edge/` - on-device inference pipeline, ROI logic, smoothing, benchmarking, and demo flow
- `backend/` - minimal FastAPI logging layer for `/update` and `/status`
- `docs/` - PRD, report outline, dataset notes, and evaluation materials

There is no frontend scope in the current PRD.

## Technical Focus

- Python
- OpenCV
- YOLOv8 / Ultralytics
- FastAPI
- SQLite
- ONNX / ONNX Runtime
- Apple Silicon MPS or CPU fallback
- local CSV/JSON logging for evidence and analysis

## Environment Setup

The project uses a shared Python virtual environment at `.venv` for both ML and edge work.

Quick start:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Or use:

```bash
make install-dev
```

Detailed setup notes live in [docs/environment-setup.md](/Users/thebkht/Projects/smart-parking-system/docs/environment-setup.md).

## Revised Goals

- fine-tune and compare YOLOv8n, YOLOv8s, and YOLOv8m on PKLot
- apply INT8 quantization and measure the accuracy and speed tradeoff
- evaluate per-weather performance and threshold-tuning behavior
- strengthen the local edge inference pipeline with ROI logic and temporal smoothing
- benchmark FPS and latency across MPS, CPU, ONNX FP32, and ONNX INT8
- capture timestamped logs that support report figures and tables

## Team Structure

- ML team (A + B) - dataset preparation, model training, export, quantization, threshold tuning, per-weather evaluation
- Edge team (C + D) - capture/inference runner, ROI logic, smoothing, benchmarking, minimal FastAPI logging, system integration

## Handoff Contract

The ML team provides:

- `best.pt`
- `best.onnx`
- documented class labels
- expected input size
- recommended confidence and IoU thresholds

The edge team consumes those artifacts and produces:

- per-spot occupancy output
- timestamped JSON or CSV logs
- minimal backend updates to `/update` and `/status`
- demo-ready inference runs on static images or live camera input

## Milestone Snapshot

- Week 4 - dataset overview, model comparison plan, training config, static-image demo, mock FastAPI endpoint
- Week 5 - train YOLOv8n and YOLOv8s, complete ROI and smoothing logic, deliver `best.pt` and `best.onnx`
- Week 6 - INT8 export, confidence and IoU sweeps, full end-to-end integration
- Week 7 - model comparison table, per-weather results, FPS and latency tables, bandwidth analysis
- Week 8 - final report and demo centered on ML gains and edge reliability

## Week 4 Run Flow

Use the Week 4 demo package for the current class milestone:

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

Then run:

```bash
python edge/detect.py --image /absolute/path/to/parking-image.jpg --post
```

More details live in [docs/week4-demo.md](/Users/thebkht/Projects/smart-parking-system/docs/week4-demo.md).

## Acceptance Criteria

- the model comparison table is complete for the trained variants
- INT8 vs FP32 results are reported
- occupancy predictions are stable enough for a live demo
- ROI-based spot classification and temporal smoothing work on the demo setup
- the final presentation does not depend on a web UI
