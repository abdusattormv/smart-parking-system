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
