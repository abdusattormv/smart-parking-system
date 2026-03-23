# Smart Parking System

An intelligent edge computing project for parking occupancy detection.

This project follows a privacy-first design:

- video stays on the edge device
- only structured JSON occupancy data is sent to the backend
- the dashboard displays live spot status and history

## Architecture

- `edge/` - Python inference service using OpenCV and YOLOv8
- `backend/` - FastAPI service with SQLite persistence
- `frontend/` - Next.js dashboard for live occupancy and analytics
- `docs/` - project notes, report outline, and demo materials

## Tech Stack

- Python
- OpenCV
- YOLOv8 / Ultralytics
- FastAPI
- SQLite + SQLAlchemy
- Next.js
- React
- Chart.js

## Project Goals

- Detect parking spot occupancy locally on a MacBook
- Send only JSON results to the backend
- Store current status and history in SQLite
- Show live spot states on a web dashboard
- Support demo-safe fallback flows for class presentation

## Suggested Workflow

1. Prepare the dataset and train the detection model in `edge/`
2. Expose backend endpoints in `backend/`
3. Build the live dashboard in `frontend/`
4. Collect architecture notes and report content in `docs/`

## Milestone Checklist

- Week 4: dataset exploration, model plan, static image demo
- Week 5: edge inference, backend persistence, initial UI
- Week 6: end-to-end live updates
- Week 7: evaluation metrics and bandwidth analysis
- Week 8: report and final presentation

## Next Steps

- scaffold the Python backend and edge environment files
- scaffold the Next.js frontend app
- add sample payloads and API contracts
- wire the end-to-end demo flow

