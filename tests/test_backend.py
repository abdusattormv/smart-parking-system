import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.main as backend_module
from backend.main import app
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    """Swap the module-level _conn for a fresh in-memory DB on each test."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    backend_module.init_db(conn)
    monkeypatch.setattr(backend_module, "_conn", conn)
    yield
    conn.close()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /update
# ---------------------------------------------------------------------------

def test_update_returns_ok():
    client = TestClient(app)
    payload = {"spot_1": "occupied", "spot_2": "free", "fps": 30.0, "confidence_avg": 0.9}
    r = client.post("/update", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["received_keys"]) == set(payload.keys())


def test_update_persists_to_db():
    client = TestClient(app)
    payload = {"spot_1": "occupied", "fps": 25.0}
    client.post("/update", json=payload)
    row = backend_module._conn.execute(
        "SELECT payload FROM updates ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    stored = json.loads(row[0])
    assert stored["spot_1"] == "occupied"


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

def test_status_no_data():
    client = TestClient(app)
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json() == {"status": "no data"}


def test_update_then_status():
    client = TestClient(app)
    payload = {"spot_1": "occupied", "spot_2": "free", "fps": 30.0, "confidence_avg": 0.9}
    client.post("/update", json=payload)

    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert data["spot_1"] == "occupied"
    assert data["spot_2"] == "free"
    assert "received_at" in data


def test_status_returns_latest():
    """GET /status should reflect the most recent POST."""
    client = TestClient(app)
    client.post("/update", json={"spot_1": "free"})
    client.post("/update", json={"spot_1": "occupied"})

    r = client.get("/status")
    assert r.json()["spot_1"] == "occupied"


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------

def test_history_empty():
    client = TestClient(app)
    r = client.get("/history")
    assert r.status_code == 200
    assert r.json() == []


def test_history_ascending_order():
    client = TestClient(app)
    payloads = [
        {"spot_1": "free",     "fps": 30.0},
        {"spot_1": "occupied", "fps": 31.0},
        {"spot_1": "free",     "fps": 32.0},
    ]
    for p in payloads:
        client.post("/update", json=p)

    r = client.get("/history")
    assert r.status_code == 200
    history = r.json()
    assert len(history) == 3
    assert history[0]["fps"] == 30.0
    assert history[1]["fps"] == 31.0
    assert history[2]["fps"] == 32.0


def test_history_limit():
    client = TestClient(app)
    for i in range(10):
        client.post("/update", json={"spot_1": "free", "fps": float(i)})

    r = client.get("/history?limit=3")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_history_each_row_has_received_at():
    client = TestClient(app)
    client.post("/update", json={"spot_1": "free", "fps": 20.0})
    history = client.get("/history").json()
    assert "received_at" in history[0]
