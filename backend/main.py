from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


app = FastAPI(title="Smart Parking System Week 4 Mock Backend")


class SpotUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/update")
def update(payload: SpotUpdate) -> Dict[str, Any]:
    data = payload.model_dump()
    return {
        "status": "ok",
        "received_keys": sorted(data.keys()),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
