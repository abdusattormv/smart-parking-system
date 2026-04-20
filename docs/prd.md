# Smart Parking System v3 PRD Summary

## Canonical Architecture

The repo is aligned to a two-stage edge pipeline:

`static camera -> fixed ROIs -> per-spot crop -> YOLOv8*-cls -> temporal smoothing -> JSON -> FastAPI`

- Stage 1 default: fixed ROI boxes from config for a static camera
- Stage 1 optional: YOLO spot detector only for experiments or future work
- Stage 2 primary: PKLot + optional CNRPark-EXT patch classification
- Main success metric: Stage 2 classification accuracy and cross-dataset generalization

## ML Scope

Stage 2 is the main workflow:

1. extract PKLot spot patches from Roboflow labels
2. optionally merge CNRPark-EXT patches
3. create `stage2_data/` train/val/test splits
4. export `pklot_test/` and `cnrpark_test/` for cross-dataset checks
5. train `yolov8{n,s,m}-cls`
6. evaluate top-1 accuracy, precision, recall, F1, confusion matrix, and per-class support

Stage 1 detection training remains available, but it is not the default repo story.

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
- threshold sweep: classifier confidence threshold used by the deployed edge path
- per-weather evaluation: only when dataset folders expose weather labels directly as `<root>/<weather>/<class>/*`

## Deliverables

- dataset outputs: `stage2_data/`, `pklot_test/`, `cnrpark_test/`
- model outputs: `runs/stage2_cls/.../weights/best.pt`
- edge runtime: [edge/detect.py](/Users/thebkht/Projects/smart-parking-system/edge/detect.py)
- backend persistence: [backend/main.py](/Users/thebkht/Projects/smart-parking-system/backend/main.py)
