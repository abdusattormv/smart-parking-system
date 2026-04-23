# Smart Parking System v3 PRD Summary

## Canonical Architecture

The repo is aligned to a two-stage edge pipeline:

`parking-space localization -> per-space crop -> YOLOv8*-cls -> temporal smoothing -> JSON -> FastAPI`

- Recommended ML path: full-frame parking-space detector -> per-slot crop -> occupancy classifier
- Deployment-specific edge demo: fixed ROI boxes from config for a static camera
- Stage 1 primary ML track: YOLO parking-space detector trained with scene holdout
- Stage 2 primary ML track: PKLot + optional CNRPark-EXT patch classification
- Single-model baseline: full-frame occupancy detector with `free` / `occupied` classes for ML comparison only
- Main success metrics: Stage 1 slot-detection generalization by held-out scene and Stage 2 occupancy accuracy
- Detection-style PKLot prep must deduplicate Roboflow variants, exclude zero-label frames, and prevent scene leakage across train/val/test

## ML Scope

The ML workflow is:

1. deduplicate PKLot Roboflow source frames by normalized base id
2. rebuild Stage 1 and single-model splits by scene holdout, not random images
3. train Stage 1 full-frame parking-space detector
4. crop Stage 2 patches from accepted slot annotations while inheriting the same scene split
5. optionally merge CNRPark-EXT patches
6. train `yolov8{n,s,m}-cls`
7. evaluate top-1 accuracy, precision, recall, F1, confusion matrix, and per-class support

Stage 1 detection training is the primary generalization track. The repo also supports an ML-only single-model occupancy detector baseline. The deployed/default repo story remains two-stage.

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

- primary split: validation from a scene-held-out dataset
- cross-dataset: PKLot -> CNRPark and CNRPark -> PKLot when both datasets are present
- model comparison: `n`, `s`, `m`
- single-model baseline comparison: full-frame occupancy detection vs the separate Stage 1 + Stage 2 approach
- threshold sweep: classifier confidence threshold used by the deployed edge path
- per-weather evaluation: only when dataset folders expose weather labels directly as `<root>/<weather>/<class>/*`
- image-level random split metrics are not sufficient evidence of full-frame generalization

## Deliverables

- dataset outputs: `stage2_data/`, `pklot_test/`, `cnrpark_test/`
- ML-only baseline outputs: `single_model_data/`, `ml/single_model.yaml`, `runs/single_model_det/.../weights/best.pt`
- model outputs: `runs/stage2_cls/.../weights/best.pt`
- edge runtime: [edge/detect.py](/Users/thebkht/Projects/smart-parking-system/edge/detect.py)
- backend persistence: [backend/main.py](/Users/thebkht/Projects/smart-parking-system/backend/main.py)
