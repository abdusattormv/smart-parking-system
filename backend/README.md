# Backend Layer

The backend is part of the current PRD, but only as a minimal logging layer for the edge pipeline.

## Current Scope

The backend team responsibility is intentionally small:

- `POST /update` receives occupancy JSON from the edge script
- `GET /status` returns the latest snapshot
- optional `GET /history` returns recent logged results for evaluation
- SQLite stores lightweight result logs only

## Week 4 Deliverable

For Week 4, only a mock `POST /update` endpoint is required. It should accept the demo payload and return a success response so the edge team can show the end-to-end flow in class.

## Not in Scope

- no frontend serving
- no WebSocket support
- no complex relational schema
- no business-heavy analytics layer

## Design Goal

Keep the service tiny, demo-safe, and easy to remove if the team decides local file logging is enough for the final presentation.
