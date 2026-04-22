# Edge Track

The edge pipeline follows the v3 architecture:

`static camera -> fixed ROIs -> crop each spot -> YOLOv8*-cls -> status smoothing -> JSON -> FastAPI`

## Default Behavior

- fixed ROI boxes are the default Stage 1 path
- Stage 2 classification is the required model path for normal use
- smoothing stabilizes only the final occupied/free status
- confidence values are reported directly from the classifier and are not smoothed
- image mode and camera mode share the same inference and postprocess logic
- camera mode also updates `logs/latest_frame.jpg` on every processed frame for backend MJPEG streaming

## Optional Stage 1 Detector

`edge/detect.py` supports a YOLO Stage 1 spot detector behind `--stage1-detector`, but that is secondary work. The main demo and main accuracy story should assume static camera placement plus configured ROIs.

## Config

Copy [edge/config.example.yaml](/Users/thebkht/Projects/smart-parking-system/edge/config.example.yaml) to `edge/config.yaml` and adjust:

- `model.stage2_path` for the classifier checkpoint
- `input.frame_interval_ms` to control default camera processing cadence
- `backend.timeout_s` and `backend.retry_delay_s` to keep camera mode responsive when the backend is slow or unavailable
- `stream.max_width` and `stream.jpeg_quality` to control default MJPEG cost
- `postprocess.classifier_threshold` for deployed occupied/free thresholding
- `postprocess.smoothing_window` for temporal status smoothing
- `rois` for camera-specific parking spot boxes

## Commands

Static image:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  --save-annotated logs/annotated.jpg
```

Live camera:

```bash
python edge/detect.py \
  --camera 0 \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  --post
```

The default camera profile is intentionally reduced now: `frame_interval_ms: 500`, `stream.max_width: 640`, `stream.jpeg_quality: 55`, `backend.timeout_s: 0.75`, and `backend.retry_delay_s: 10.0`.

Override the streamed frame target if needed:

```bash
python edge/detect.py \
  --camera 0 \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  --latest-frame-path logs/latest_frame.jpg
```

If the live stream feels laggy, lower the stream cost without changing inference:

```bash
python edge/detect.py \
  --camera 0 \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  --stream-max-width 640 \
  --stream-jpeg-quality 55
```

macOS iPhone camera:

```bash
python edge/detect.py \
  --camera iphone \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt \
  --post
```

`--camera iphone` is macOS-only and expects an available Continuity Camera / iPhone camera device.

Optional Stage 1 detector:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage1-detector \
  --stage1-model runs/stage1_det/yolov8n_stage1/weights/best.pt \
  --stage2-model runs/stage2_cls/yolov8n_stage2/weights/best.pt
```

## Payload Contract

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
