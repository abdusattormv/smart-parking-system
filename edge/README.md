# Edge Track

Python-based on-device inference pipeline for local parking occupancy detection.

## Purpose

The edge track is responsible for running the parking model locally, converting detections into per-spot occupancy states, stabilizing those states across frames, and recording local evidence for evaluation and demos.

## Main Responsibilities

- capture frames from a webcam or load static parking images
- load the trained model artifact (`best.pt` or optional `best.onnx`)
- run inference on `mps` when available with `cpu` fallback
- map detections to predefined parking spot regions
- apply temporal smoothing or debouncing to reduce flicker
- output per-spot occupied/free predictions
- log timestamps, statuses, confidences, FPS, and latency in local JSON or CSV files

## Week 4 Deliverable

Week 4 only requires a stable static-image demo. The first implementation should:

- run pretrained YOLO on one parking image
- print detections in the terminal
- emit the agreed JSON contract shape
- optionally POST that payload to the mock FastAPI backend

ROI-based occupancy and temporal smoothing are intentionally deferred to Week 5.

## Ownership Split

- Edge Engineer 1 - capture/inference runner, model loading, device handling, FPS and latency measurement
- Edge Engineer 2 - spot mapping, post-processing, local logging, and demo integration

## Expected Inputs

From the ML track:

- final trained model artifact
- class labels
- expected input size
- recommended confidence threshold
- recommended IoU threshold

## Expected Outputs

- per-spot occupancy predictions
- timestamped local logs for analysis and report evidence
- stable demo runs for static-image and live-camera modes

## Validation Checklist

- static image mode works correctly
- live camera mode works correctly
- `mps` and `cpu` paths are handled safely
- smoothing behavior is stable across consecutive frames
- logs are usable for Week 7 analysis and Week 8 reporting
