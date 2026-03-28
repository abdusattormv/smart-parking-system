# Smart Parking System — Full Project PRD

**Intelligent Edge Computing · Weeks 3–8 · 4-Person Team**

---

## Project metadata

| Field | Value |
|---|---|
| Project | Smart Parking System — Intelligent Edge Computing |
| Duration | 6 weeks (Week 3 → Week 8) |
| Current week | Week 5 — Training + edge refinement |
| Edge device | MacBook (Apple Silicon MPS / Intel CPU) |
| Inference stack | Python · OpenCV · YOLOv8 (ultralytics) · requests |
| Backend | Python · FastAPI · SQLite (minimal — result logging only) |
| Frontend | None — dropped in favour of deeper ML work |
| Team split | ML team (A + B) · Edge team (C + D) |
| Final deadline | Week 8 — technical report (email) + presentation (class) |

---

## 1. Problem statement

Finding an available parking spot in a busy lot is inefficient: drivers waste time circling, fuel is wasted, and the lot operator has no real-time occupancy data. Existing solutions either require expensive in-ground sensors per spot, or stream video to the cloud, which raises bandwidth costs and privacy concerns.

This project builds a privacy-preserving, low-cost alternative: a camera on a laptop runs AI inference locally, classifies each spot as free or occupied, and transmits only a small JSON payload to a backend. No raw video ever leaves the device.

| Problem | Impact |
|---|---|
| No real-time occupancy visibility | Drivers waste time and fuel searching |
| Cloud video streaming is expensive | High bandwidth cost and privacy risk |
| Sensor-per-spot hardware is costly | Not viable for small or temporary lots |
| No analytics on usage patterns | Operators cannot optimise pricing or layout |

## 2. Scope change — professor feedback

The professor recommended shifting focus from a web dashboard to deeper ML work. This is the right call for an edge computing course: the academically interesting contributions are in the inference pipeline, not the UI.

### What changed

| Before | After |
|---|---|
| 4 separate roles (ML, Edge, Backend, Frontend) | 2 teams of 2 (ML team + Edge team) |
| Next.js dashboard + WebSocket + Chart.js | Dropped entirely |
| FastAPI with full DB schema + live push | Minimal FastAPI — POST /update + GET /status only |
| One model (YOLOv8n) | Compare YOLOv8n / s / m + INT8 quantization |
| Generic evaluation | Per-weather evaluation, PR curves, quantization analysis |

### Why this is better

A model comparison table (YOLOv8n vs s vs m: mAP vs FPS vs size) combined with a quantization experiment (FP32 vs INT8) directly demonstrates understanding of the core edge computing constraint — operating under resource limits. That is more valuable for this course than a React dashboard.

## 3. Goals and non-goals

### Goals

- Fine-tune and compare YOLOv8n, YOLOv8s, and YOLOv8m on PKLot
- Apply INT8 quantization and measure accuracy/speed tradeoff
- Evaluate per weather condition (sunny, cloudy, rainy)
- Run full inference pipeline on-device with no raw video transmitted
- Tune confidence threshold and NMS for a parking-specific operating point
- Add ROI-based spot detection and temporal smoothing
- Benchmark FPS and latency across MPS, CPU, and ONNX INT8 backends
- Measure bandwidth savings vs equivalent video stream
- Produce a technical report and final presentation by Week 8

### Non-goals

- Web dashboard or any frontend UI
- WebSocket or real-time push infrastructure
- Mobile app
- Hardware beyond MacBook
- Real-time model retraining or active learning
- Production deployment

## 4. Team structure

### ML team — Members A + B

**Focus:** Everything from raw dataset to trained, evaluated model.

| Task | Description |
|---|---|
| Dataset preparation | Download PKLot, convert XML to YOLO `.txt`, write `data.yaml`, apply 70/15/15 split |
| Model training | Fine-tune YOLOv8n, YOLOv8s, YOLOv8m on PKLot |
| Model export | Export best checkpoint as `best.pt` and `best.onnx` |
| Quantization | INT8 quantization via ONNX and measure mAP drop vs speed gain |
| Threshold tuning | Sweep confidence and IoU NMS to find the operating point |
| Per-weather eval | Evaluate mAP separately on sunny, cloudy, and rainy splits |
| Final evaluation | mAP@50, mAP@50-95, precision, recall, F1, confusion matrix, PR curve |
| Report sections | Dataset, model architecture, training, results, evaluation |

### Edge team — Members C + D

**Focus:** Everything from model artifact to live inference.

| Task | Description |
|---|---|
| `detect.py` | OpenCV capture, YOLO inference loop, spot classification, JSON output |
| ROI regions | Define fixed polygons per spot and use overlap-based occupancy check |
| Temporal smoothing | Majority vote over last N frames to reduce flicker |
| FPS benchmarks | Measure speed on MPS, CPU, ONNX FP32, ONNX INT8 |
| FastAPI (minimal) | `POST /update` receives JSON and `GET /status` returns the latest state |
| Bandwidth analysis | Log payload sizes and compare JSON with H.264 video |
| System integration | Load trained model from the ML team and run the full pipeline end-to-end |
| Report sections | System architecture, edge pipeline, FPS results, bandwidth analysis |

### Team handoff

The ML team delivers `best.pt` and `best.onnx` at the end of Week 5. The edge team loads them into `detect.py` in Week 6. Before that handoff, the edge team can develop against generic pretrained YOLOv8 weights.

### Agreed JSON contract

```json
{
  "spot_1": "free",
  "spot_2": "occupied",
  "spot_3": "free",
  "spot_4": "occupied",
  "confidence_avg": 0.94,
  "fps": 28.3,
  "timestamp": "2026-03-29T14:32:01Z"
}
```

## 5. System architecture

Three layers, all processing local, only JSON transmitted.

```text
Webcam / images
  -> OpenCV
  -> YOLOv8 (MPS / CPU / ONNX)
  -> ROI match
  -> temporal smoother
  -> spot status dict
  -> requests.post()
  -> minimal FastAPI logger
  -> SQLite current/history view
```

Why this qualifies as edge computing:

- AI inference runs entirely on the local device
- raw video never leaves the laptop
- only structured results are transmitted
- the device is the intelligence, not a dumb camera feeding the cloud

## 6. Dataset — PKLot

| Property | Value |
|---|---|
| Source | Pontifical Catholic University of Parana, Brazil |
| Download | Kaggle (`ammarnassanalhajali/pklot-dataset`) or Roboflow / public mirrors |
| Total images | 12,417 full overhead parking lot frames |
| Labelled spots | 695,899 individual spot annotations |
| Parking lots | 3 — PUCPR, UFPR04, UFPR05 |
| Classes | 2 — `occupied`, `empty` |
| Weather conditions | Sunny, cloudy, rainy |
| Annotation format | XML bounding boxes converted to YOLO `.txt` |
| License | CC Attribution 4.0 |

### Data split

| Split | Ratio | Images |
|---|---|---|
| Train | 70% | ~8,692 |
| Validation | 15% | ~1,863 |
| Test | 15% | ~1,862 |

## 7. ML model

### Model comparison plan

| Model | Parameters | GFLOPs | Expected mAP@50 | Expected FPS (MPS) |
|---|---|---|---|---|
| YOLOv8n | 3.2M | 8.7 | ~96% | 50–60 |
| YOLOv8s | 11.2M | 28.6 | ~97% | 30–40 |
| YOLOv8m | 25.9M | 78.9 | ~98% | 15–25 |

### Training configuration

```yaml
model: yolov8n.pt
data: data.yaml
epochs: 50
imgsz: 640
batch: 16
optimizer: AdamW
lr0: 0.01
patience: 10
device: mps
project: runs/parking
name: yolov8n_pklot
```

### Export

```python
from ultralytics import YOLO

model = YOLO("runs/parking/yolov8n_pklot/weights/best.pt")
model.export(format="onnx")
model.export(format="onnx", int8=True)
```

## 8. ML enhancements

- INT8 quantization: compare FP32 vs INT8 model size, mAP, and FPS
- Per-weather evaluation: sunny, cloudy, rainy performance split
- Confidence threshold sweep: tune precision vs recall for parking use
- NMS IoU tuning: reduce double-counting across tightly packed spots

## 9. Edge inference pipeline

- `detect.py` handles capture, model inference, ROI-based spot checks, smoothing, and JSON output
- ROI-based occupancy is determined from overlap with predefined parking spot regions
- Temporal smoothing uses a majority vote over the last N frames

## 10. FastAPI backend (minimal)

The backend exists only to receive and log edge results and expose the latest snapshot for evaluation. It intentionally stays tiny:

- `POST /update`
- `GET /status`
- optional `GET /history`

## 11. Evaluation plan

### Model accuracy

- mAP@50
- mAP@50-95
- precision
- recall
- F1
- confusion matrix
- PR curve
- per-weather mAP
- FP32 vs INT8 comparison

### Inference speed

- benchmark MPS, CPU, ONNX FP32, and ONNX INT8
- report FPS, latency, and model size

### Bandwidth analysis

- measure real JSON payload sizes
- compare with a conservative H.264 parking camera stream
- report percentage savings

## 12. Week-by-week delivery plan

- Week 4: completed - dataset overview, model comparison plan, training config, `detect.py` skeleton, mock FastAPI, live static-image demo
- Week 5: train YOLOv8n and YOLOv8s, finish ROI and smoothing logic, deliver `best.pt` and `best.onnx`
- Week 6: optional YOLOv8m, INT8 export, confidence and IoU sweeps, full end-to-end integration
- Week 7: final evaluation tables and figures, FPS and latency benchmarks, system stability test
- Week 8: finish report, present results, run live demo, submit report

## 13. Acceptance criteria

- Week 4: completed - dataset and training plan were presented, the static-image demo package was implemented, and all four members have a defined speaking role
- Week 5–6: trained model loads on the MacBook, ROI and smoothing work, full local pipeline runs
- Week 7: model comparison table, INT8 comparison, per-weather results, and bandwidth analysis are complete
- Week 8: report is submitted and the live demo works in class

## 13A. Current project status after Week 4

- PRD updated to the ML-team and Edge-team structure
- Week 4 demo package implemented
- `edge/detect.py` added for static-image inference with terminal JSON output
- `backend/main.py` added for mock `POST /update`
- dataset and model comparison notes documented for presentation use
- ROI logic, smoothing, training runs, ONNX benchmarking, and SQLite logging remain Week 5+ work

## 14. Technical report outline

1. Abstract
2. Introduction
3. Related work
4. System architecture
5. Dataset
6. ML model
7. Model comparison
8. Quantization
9. Edge inference pipeline
10. Evaluation
11. Discussion
12. Conclusion
13. References

## 15. Risks and mitigations

| Severity | Risk | Mitigation |
|---|---|---|
| High | Demo crashes during Week 4 presentation | Pre-record a fallback clip and keep screenshots |
| High | YOLOv8s/m training finishes late | Train `n` first and use Colab for heavier variants |
| Medium | INT8 quantization causes noticeable mAP loss | Report it as a valid tradeoff result |
| Medium | Intel MacBook lacks MPS | Use YOLOv8n and ONNX INT8 for the demo |
| Medium | PKLot download is blocked | Use alternate mirrors or share via Drive |
| Medium | ML handoff arrives late | Edge team develops against pretrained weights first |
| Low | JSON contract mismatch | Lock the payload shape early and validate it |
| Low | FastAPI becomes unnecessary | Keep it optional and tiny |
