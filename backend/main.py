from __future__ import annotations

import json
import sqlite3
from asyncio import sleep
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()
DB = Path("parking.db")
LATEST_FRAME_PATH = Path("logs/latest_frame.jpg")
_conn: sqlite3.Connection | None = None


class ParkingUpdate(BaseModel):
    spots: Dict[str, Literal["occupied", "free"]] = Field(default_factory=dict)
    confidence: Dict[str, float] = Field(default_factory=dict)
    timestamp: str


class HistoryItem(BaseModel):
    payload: ParkingUpdate
    recorded_at: str


class HistoryResponse(BaseModel):
    items: List[HistoryItem]


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    global _conn
    if conn is None:
        conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            recorded TEXT NOT NULL
        )
        """
    )
    conn.commit()
    _conn = conn
    return conn


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = init_db()
    return _conn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def generate_mjpeg_stream(
    frame_path: Path = LATEST_FRAME_PATH,
    poll_interval: float = 0.1,
):
    last_signature: tuple[int, int] | None = None
    while True:
        try:
            stat = frame_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                frame_bytes = frame_path.read_bytes()
                if frame_bytes:
                    last_signature = signature
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
        except FileNotFoundError:
            pass
        await sleep(poll_interval)


init_db()


@app.post("/update")
async def update(payload: ParkingUpdate) -> dict:
    conn = get_conn()
    conn.execute(
        "INSERT INTO log (payload, recorded) VALUES (?, ?)",
        (payload.model_dump_json(), utc_now_iso()),
    )
    conn.commit()
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    conn = get_conn()
    row = conn.execute("SELECT payload FROM log ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(row[0]) if row else {}


@app.get("/history")
async def history(limit: int = 100) -> HistoryResponse:
    conn = get_conn()
    rows = conn.execute(
        "SELECT payload, recorded FROM log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    items = [
        HistoryItem(payload=ParkingUpdate(**json.loads(payload)), recorded_at=recorded)
        for payload, recorded in rows
    ]
    return HistoryResponse(items=items)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
