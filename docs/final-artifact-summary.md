# Final Artifact Summary

## Dataset Inventory

### Stage 1

| Split | Images | Boxes | Scenes |
| --- | ---: | ---: | ---: |
| train | 1953 | 72607 | 123 |
| val | 337 | 13433 | 27 |
| test | 396 | 13733 | 27 |

### Stage 2

| Split | Free | Occupied |
| --- | ---: | ---: |
| train | 47978 | 62230 |
| val | 10383 | 13538 |
| test | 10291 | 13513 |

### Cross-Dataset Exports

- `pklot_test`: present=True, free=221, occupied=808
- `cnrpark_test`: present=True, free=9849, occupied=11897

### Weather Export

- `stage2_weather`: present=True splits={'sunny': {'free': 25665, 'occupied': 37513}, 'cloudy': {'free': 21067, 'occupied': 23176}, 'rainy': {'free': 18926, 'occupied': 18618}}

## Checkpoints

- Stage 1 `yolov8s`: present=True path=`runs/stage1_det/yolov8s_stage1/weights/best.pt`
- Stage 1 `yolov8m`: present=False path=`runs/stage1_det/yolov8m_stage1/weights/best.pt`
- Stage 2 `yolov8n-cls`: present=True path=`runs/stage2_cls/yolov8n_stage2/weights/best.pt`
- Stage 2 `yolov8s-cls`: present=False path=`runs/stage2_cls/yolov8s_stage2/weights/best.pt`
- Stage 2 `yolov8m-cls`: present=False path=`runs/stage2_cls/yolov8m_stage2/weights/best.pt`

## Acceptance Checks

- stage1_detector_checkpoint: PASS
- stage2_n_checkpoint: PASS
- stage2_s_checkpoint: MISSING
- stage2_m_checkpoint: MISSING
- stage1_eval_table: PASS
- stage2_eval_table: PASS
- stage2_model_comparison: MISSING
- threshold_sweep: MISSING
- cross_dataset_eval: PASS
- per_weather_eval: MISSING
- benchmark_results: PASS
- bandwidth_report: PASS
- stability_summary: MISSING

## Latest Metrics Snapshot

- Stage 1 evaluation: `{"mAP50": "0.98751", "mAP50_95": "0.71444", "model": "yolov8s_stage1", "precision": "0.95984", "recall": "0.95027", "scene_count": "27", "scene_leakage": "False", "split": "val"}`
- Stage 2 evaluation: `{"confusion_matrix": "[[266, 1], [4, 817]]", "dataset": "stage2_data/val", "f1": "0.9969", "model": "yolov8n_stage2", "precision": "0.9988", "recall": "0.9951", "sample_count": "1088", "support_free": "267", "support_occupied": "821", "threshold": "0.5", "top1_accuracy": "0.9954"}`
- Stage 2 model comparison: `missing`
- Stage 2 threshold sweep: `missing`
- Stage 2 cross-dataset: `{"confusion_matrix": "[[9583, 266], [1946, 9951]]", "dataset": "cnrpark_test", "f1": "0.9", "model": "yolov8n_stage2", "precision": "0.974", "recall": "0.8364", "sample_count": "21746", "support_free": "9849", "support_occupied": "11897", "threshold": "0.5", "top1_accuracy": "0.8983"}`
- Stage 2 per-weather: `missing`
