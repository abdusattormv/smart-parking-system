# Backend Layer

The backend is a minimal persistence layer for the v3 edge payload. It does not add frontend concerns or legacy summary fields.

## Endpoints

- `POST /update` accepts the v3 payload and stores it as-is
- `GET /status` returns the latest stored payload as-is
- `GET /history` returns stored payloads in a stable wrapper with `recorded_at`
- `GET /health` returns a basic health response
- `GET /stream` returns the latest annotated edge frame as an MJPEG stream

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

## Live Stream

`GET /stream` serves `logs/latest_frame.jpg` as a multipart MJPEG response for browser or VLC clients.

Example:

```html
<img src="http://127.0.0.1:8000/stream" alt="Parking stream">
```

## History Shape

```json
{
  "items": [
    {
      "payload": {
        "spots": {
          "spot_1": "free"
        },
        "confidence": {
          "spot_1": 0.91
        },
        "timestamp": "2026-04-21T00:00:00Z"
      },
      "recorded_at": "2026-04-21T00:00:01Z"
    }
  ]
}
```
