# Week 4 ML Notes

These notes support the Week 4 presentation and keep the scope aligned with the PRD.

## Dataset Summary

- Dataset: PKLot
- Source: Pontifical Catholic University of Parana, Brazil
- Classes: `empty`, `occupied`
- Parking lots: PUCPR, UFPR04, UFPR05
- Weather conditions: sunny, cloudy, rainy
- Annotation source format: XML bounding boxes
- Planned training format: YOLO `.txt`

## Practical Week 4 Dataset Assumption

For Week 4, the team can use a public mirror or local copy of PKLot. The live demo does not require the full dataset or a trained PKLot model as long as a sample parking image is available.

## Planned Split

- Train: 70%
- Validation: 15%
- Test: 15%

The test split must stay held out for Week 7 evaluation.

## Model Comparison Plan

| Model | Parameters | GFLOPs | Expected mAP@50 | Expected FPS (MPS) |
|---|---|---|---|---|
| YOLOv8n | 3.2M | 8.7 | ~96% | 50-60 |
| YOLOv8s | 11.2M | 28.6 | ~97% | 30-40 |
| YOLOv8m | 25.9M | 78.9 | ~98% | 15-25 |

## Why These Three Models

- `yolov8n` gives the fastest edge baseline
- `yolov8s` gives a middle tradeoff between speed and accuracy
- `yolov8m` tests whether a heavier model is still practical on a MacBook

This comparison is the core Week 4 ML story because it frames the accuracy/latency tradeoff that the later evaluation will measure.

## Training Configuration For Week 5

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

Repeat the same setup for `yolov8s.pt` and `yolov8m.pt`.

## Export Plan

- `best.pt` for direct ultralytics inference
- `best.onnx` for the edge team and later ONNX benchmarking
- INT8 export is deferred to Week 6

## Week 4 Completion Checklist

- dataset named and sourced
- classes and weather conditions explained
- train/val/test split defined
- YOLOv8n/s/m comparison table ready
- training config ready
- export plan ready
