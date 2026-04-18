from fastapi import FastAPI
from datetime import datetime
import sqlite3
import json

app = FastAPI()
DB = "parking.db"


def init_db() -> None:
    con = sqlite3.connect(DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS log (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            payload  TEXT,
            recorded TEXT
        )
        """
    )
    con.commit()
    con.close()


init_db()


@app.post("/update")
async def update(payload: dict) -> dict:
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO log (payload, recorded) VALUES (?, ?)",
        (json.dumps(payload), datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    con = sqlite3.connect(DB)
    row = con.execute(
        "SELECT payload FROM log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    con.close()
    return json.loads(row[0]) if row else {}


@app.get("/history")
async def history(limit: int = 100) -> list:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT payload, recorded FROM log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return [{"data": json.loads(r[0]), "time": r[1]} for r in rows]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
