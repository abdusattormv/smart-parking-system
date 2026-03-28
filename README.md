# Smart Parking System

An intelligent edge computing project for parking occupancy detection with a stronger focus on ML quality and on-device reliability.

## Project Story

This project is a camera-based smart parking prototype that improves parking occupancy detection accuracy and robustness on-device, then demonstrates a stable edge inference pipeline with lightweight local logging.

The system follows a privacy-first design:

- video stays on the edge device
- inference runs locally on a MacBook
- results are recorded as lightweight local logs for evaluation
- a full web dashboard is deferred in favor of ML and edge improvements

## Core Tracks

- `edge/` - on-device inference pipeline, spot mapping, smoothing, logging, and demo flow
- `docs/` - revised PRD, report outline, dataset notes, and evaluation materials

Supporting folders remain in the repo for future extension, but they are not part of the core deliverables for this revised scope:

- `backend/` - optional lightweight support layer or future integration work
- `frontend/` - deferred dashboard ideas and future extension notes

## Technical Focus

- Python
- OpenCV
- YOLOv8 / Ultralytics
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

Detailed setup notes live in [docs/environment-setup.md](https://github.com/thebkht/smart-parking-system/blob/main/docs/README.md).

## Revised Goals

- improve parking occupancy detection accuracy and robustness
- strengthen the local edge inference pipeline for repeatable demos
- evaluate model quality with mAP@50, precision, recall, and F1
- evaluate edge behavior with FPS, latency, and output stability
- capture timestamped local logs that support report figures and tables

## Team Structure

- ML Engineer 1 - dataset preparation, annotation conversion, augmentation, split quality
- ML Engineer 2 - training, tuning, evaluation, error analysis, final model artifact
- Edge Engineer 1 - capture/inference runner, model loading, device handling, FPS and latency measurement
- Edge Engineer 2 - spot mapping, temporal smoothing, local logging, demo integration

## Handoff Contract

The ML track provides:

- `best.pt`
- optional `best.onnx`
- documented class labels
- expected input size
- recommended confidence and IoU thresholds

The edge track consumes those artifacts and produces:

- per-spot occupancy output
- timestamped local JSON or CSV logs
- demo-ready inference runs on static images or live camera input

## Milestone Snapshot

- Week 4 - revised scope, dataset analysis, ML improvement plan, static-image demo
- Week 5 - cleaned dataset pipeline, tuned training baseline, trained model wired into edge script
- Week 6 - full local pipeline from capture/image to inference, smoothing, and log output
- Week 7 - final ML metrics, robustness analysis, FPS/latency results, report figures
- Week 8 - final report and demo centered on ML gains and edge reliability

## Acceptance Criteria

- the final model clearly improves on the initial baseline
- occupancy predictions are stable enough for a live demo
- local logs are usable as evaluation evidence
- each member has a distinct contribution within ML or edge work
- the final presentation does not depend on a web UI
