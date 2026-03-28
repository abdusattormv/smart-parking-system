# Smart Parking System PRD Revision: ML + Edge Focus

## Summary

Revise the project from a 4-layer "ML + backend + frontend dashboard" system into a 2-track intelligent edge project where the main value is stronger parking-occupancy ML and a reliable on-device edge pipeline.

The new project story is:

> A camera-based smart parking prototype that improves occupancy detection accuracy and robustness on-device, then demonstrates a stable edge inference pipeline with lightweight local logging.

Any backend or frontend work is optional support only and is not part of the core deliverables.

## Revised Architecture

### ML Track

Focus areas:

- dataset preparation
- preprocessing and augmentation
- model training and tuning
- evaluation and error analysis

### Edge Track

Focus areas:

- frame capture
- inference runner
- spot-level occupancy logic
- temporal smoothing
- local logging
- demo workflow

## Goals

- improve parking occupancy detection accuracy and robustness
- strengthen the local inference pipeline for reliable demos
- evaluate model quality with mAP@50, precision, recall, and F1
- evaluate edge behavior with FPS, latency, and output stability
- record local logs that support the report and presentation

## Non-Goals

- a production deployment
- multi-camera coordination
- cloud video streaming
- a full web dashboard
- a required public backend API

Deferred note:

> A full web dashboard is deferred; this version focuses on ML quality and edge reliability.

## Interfaces and Ownership

### ML Engineer 1

- dataset preparation
- annotation conversion and verification
- augmentation pipeline
- training split quality checks

### ML Engineer 2

- model training
- hyperparameter tuning
- evaluation metrics
- error analysis
- final model artifact

### Edge Engineer 1

- capture and inference runner
- model loading
- `mps` and `cpu` device handling
- FPS and latency measurement

### Edge Engineer 2

- spot mapping
- temporal smoothing and post-processing
- local logging
- demo script integration

## ML-to-Edge Handoff

The ML track provides:

- `best.pt`
- optional `best.onnx`
- class labels
- expected input size
- recommended confidence threshold
- recommended IoU threshold

The edge track provides:

- per-spot occupancy output
- timestamped JSON or CSV logs
- repeatable demo runs on static-image or live-camera input

## Success Metrics

### Accuracy and Robustness

- mAP@50
- precision
- recall
- F1
- false positives and false negatives by condition

### Edge Reliability

- FPS
- end-to-end latency
- stability during repeated runs
- completeness of local logs

## Milestones

### Week 4

- present the revised scope and justify dropping dashboard emphasis
- show dataset characteristics and parking conditions
- explain the ML improvement plan
- show a basic static-image inference demo

### Week 5

- complete cleaned dataset preparation
- run the first tuned training cycle
- produce baseline metrics and identify common failure cases
- run inference from the trained artifact inside the edge script

### Week 6

- complete the full local pipeline from capture or image input to logged occupancy output
- integrate spot logic and temporal smoothing
- demonstrate stable repeated runs without crashes

### Week 7

- report final ML metrics and robustness findings
- report FPS, latency, and reliability observations
- prepare figures and tables for the report and presentation

### Week 8

- deliver the final report
- deliver the final demo centered on ML gains and edge reliability

## Acceptance Criteria

- the final model beats or clearly improves on the initial baseline
- occupancy predictions are stable enough for a live demo
- local logging captures evidence for evaluation
- each member has a distinct technical contribution tied to ML or edge
- the final presentation does not depend on a web UI

## Test Plan

### Dataset Checks

- verify train/val/test split integrity
- verify label conversion correctness on sampled images

### ML Evaluation

- compare baseline and tuned models on the same test split
- report mAP@50, precision, recall, and F1
- include failure cases for shadows, occlusion, and lighting variation

### Edge Validation

- test static image mode
- test live camera mode
- test `mps` and `cpu` fallback behavior
- verify output format for per-spot occupancy
- verify smoothing behavior across consecutive frames
- verify logs are usable for tables and plots

### Demo Readiness

- rehearse with a fallback recorded clip or saved sample images
- confirm the demo can run without network or UI dependencies

## Assumptions

- the revised PRD replaces the old dashboard-heavy narrative
- "enhance the ML" means better accuracy and robustness, not more product features
- backend and frontend remain optional future extensions
- local logging is sufficient support infrastructure for current project goals
