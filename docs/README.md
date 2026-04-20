# Docs

Use [docs/prd.md](/Users/thebkht/Projects/smart-parking-system/docs/prd.md) as the canonical project definition.

Supporting docs:

- [docs/week4-demo.md](/Users/thebkht/Projects/smart-parking-system/docs/week4-demo.md) for the current static-image demo flow
- [docs/week4-ml-notes.md](/Users/thebkht/Projects/smart-parking-system/docs/week4-ml-notes.md) for Stage 2 ML prep and comparison notes
- [edge/README.md](/Users/thebkht/Projects/smart-parking-system/edge/README.md) for the edge runtime contract
- [backend/README.md](/Users/thebkht/Projects/smart-parking-system/backend/README.md) for backend payload persistence

Everything in this folder should describe the same v3 architecture:

`static camera -> fixed ROIs -> per-spot crop -> classifier -> temporal smoothing -> JSON -> FastAPI`
