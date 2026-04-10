import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


DB_PATH = "parking.db"
_conn: sqlite3.Connection = None  # replaced in tests via monkeypatch


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS updates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT NOT NULL,
            payload     TEXT NOT NULL
        )
        """
    )
    conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conn
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    init_db(_conn)
    yield


app = FastAPI(title="Smart Parking System Backend", lifespan=lifespan)


class SpotUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/update")
def update(payload: SpotUpdate) -> Dict[str, Any]:
    data = payload.model_dump()
    received_at = datetime.now(timezone.utc).isoformat()
    _conn.execute(
        "INSERT INTO updates (received_at, payload) VALUES (?, ?)",
        (received_at, json.dumps(data)),
    )
    _conn.commit()
    return {
        "status": "ok",
        "received_keys": sorted(data.keys()),
        "received_at": received_at,
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    row = _conn.execute(
        "SELECT payload, received_at FROM updates ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"status": "no data"}
    return {**json.loads(row[0]), "received_at": row[1]}


@app.get("/history")
def history(limit: int = 100) -> List[Dict[str, Any]]:
    rows = _conn.execute(
        "SELECT payload, received_at FROM updates ORDER BY id ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{**json.loads(row[0]), "received_at": row[1]} for row in rows]
