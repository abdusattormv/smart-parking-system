# Week 4 Demo Runbook

This runbook covers the presentation-ready Week 4 deliverables:

- static-image YOLO demo
- terminal JSON payload in the agreed contract shape
- minimal FastAPI mock backend

## Deliverables

- `backend/main.py` - mock FastAPI app with `POST /update`
- `edge/detect.py` - static-image inference demo using pretrained YOLO
- `docs/week4-ml-notes.md` - dataset and training-plan notes for the ML team

## Demo Assumptions

- The Week 4 demo uses pretrained YOLO weights, not a PKLot-trained checkpoint.
- The live presentation uses a static parking image for stability.
- ROI-based occupancy logic, temporal smoothing, ONNX runtime, and SQLite logging are deferred to Week 5+.

## Start The Backend

From the project root:

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

Expected endpoint for the demo:

- `POST http://127.0.0.1:8000/update`

Optional health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run The Static-Image Demo

Use any parking lot image available locally:

```bash
source .venv/bin/activate
python edge/detect.py \
  --image /absolute/path/to/parking-image.jpg \
  --post \
  --save-annotated logs/week4-demo-annotated.jpg
```

If the backend is not running, omit `--post`:

```bash
python edge/detect.py --image /absolute/path/to/parking-image.jpg
```

## What To Show In Class

1. The parking image being processed
2. Bounding-box detections printed in the terminal
3. The Week 4 JSON payload with:
   - `spot_1 ... spot_n`
   - `confidence_avg`
   - `fps`
   - `timestamp`
4. Optional backend response `{"status": "ok", ...}`

## Suggested Speaking Split

- A: problem statement and why edge computing matters
- B: PKLot dataset and YOLOv8n/s/m comparison plan
- C: `detect.py` static-image demo and JSON contract
- D: minimal FastAPI mock flow and Week 5 handoff path

## Fallback Plan

- Keep one known-good parking image ready locally
- Save an annotated output image before class
- Keep a screenshot of terminal JSON in case the live run misbehaves
