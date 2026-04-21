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
| train | 1009 | 3358 |
| val | 267 | 821 |
| test | 221 | 808 |

### Cross-Dataset Exports

- `pklot_test`: present=True, free=221, occupied=808
- `cnrpark_test`: present=False, free=0, occupied=0

## Checkpoints

- Stage 1 `yolov8s`: present=False path=`runs/stage1_det/yolov8s_stage1/weights/best.pt`
- Stage 1 `yolov8m`: present=False path=`runs/stage1_det/yolov8m_stage1/weights/best.pt`
- Stage 2 `yolov8n-cls`: present=False path=`runs/stage2_cls/yolov8n_stage2/weights/best.pt`
- Stage 2 `yolov8s-cls`: present=False path=`runs/stage2_cls/yolov8s_stage2/weights/best.pt`
- Stage 2 `yolov8m-cls`: present=False path=`runs/stage2_cls/yolov8m_stage2/weights/best.pt`

## Acceptance Checks

- stage1_detector_checkpoint: MISSING
- stage2_n_checkpoint: MISSING
- stage2_s_checkpoint: MISSING
- stage2_m_checkpoint: MISSING
- stage1_eval_table: MISSING
- stage2_eval_table: MISSING
- stage2_model_comparison: MISSING
- threshold_sweep: MISSING
- benchmark_results: PASS
- bandwidth_report: PASS
- stability_summary: MISSING

## Latest Metrics Snapshot

- Stage 1 evaluation: `missing`
- Stage 2 evaluation: `missing`
- Stage 2 model comparison: `missing`
- Stage 2 threshold sweep: `missing`
