# Smart Parking System v3 PRD Summary

## Canonical Architecture

The repo is aligned to a two-stage edge pipeline:

`static camera -> fixed ROIs -> per-spot crop -> YOLOv8*-cls -> temporal smoothing -> JSON -> FastAPI`

- Stage 1 default: fixed ROI boxes from config for a static camera
- Stage 1 optional: YOLO spot detector only for experiments or future work
- Stage 2 primary: PKLot + optional CNRPark-EXT patch classification
- Single-model baseline: full-frame occupancy detector with `free` / `occupied` classes for ML comparison only
- Main success metric: Stage 2 classification accuracy and cross-dataset generalization
- Detection-style PKLot prep must exclude zero-label frames because they are unsafe supervision for full-frame detection baselines

## ML Scope

Stage 2 is the main workflow:

1. extract PKLot spot patches from Roboflow labels
2. optionally merge CNRPark-EXT patches
3. create `stage2_data/` train/val/test splits
4. export `pklot_test/` and `cnrpark_test/` for cross-dataset checks
5. train `yolov8{n,s,m}-cls`
6. evaluate top-1 accuracy, precision, recall, F1, confusion matrix, and per-class support

Stage 1 detection training remains available, and the repo also supports an ML-only single-model occupancy detector baseline. The deployed/default repo story remains two-stage.

## Runtime Contract

Edge output and backend storage use the same payload:

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

- `confidence` is keyed by spot id
- `timestamp` is UTC
- smoothing changes only the status in `spots`
- smoothing does not rewrite classifier confidence

## Evaluation Expectations

- primary split: validation
- cross-dataset: PKLot -> CNRPark and CNRPark -> PKLot when both datasets are present
- model comparison: `n`, `s`, `m`
- single-model baseline comparison: full-frame occupancy detection vs the separate Stage 1 + Stage 2 approach
- threshold sweep: classifier confidence threshold used by the deployed edge path
- per-weather evaluation: only when dataset folders expose weather labels directly as `<root>/<weather>/<class>/*`

## Deliverables

- dataset outputs: `stage2_data/`, `pklot_test/`, `cnrpark_test/`
- ML-only baseline outputs: `single_model_data/`, `ml/single_model.yaml`, `runs/single_model_det/.../weights/best.pt`
- model outputs: `runs/stage2_cls/.../weights/best.pt`
- edge runtime: [edge/detect.py](/Users/thebkht/Projects/smart-parking-system/edge/detect.py)
- backend persistence: [backend/main.py](/Users/thebkht/Projects/smart-parking-system/backend/main.py)
