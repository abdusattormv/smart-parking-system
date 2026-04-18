PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

VARIANT ?= n
DEVICE  ?= mps
PKLOT_DIR ?= datasets/pklot_raw
WEIGHTS ?= runs/parking/yolov8$(VARIANT)_pklot/weights/best.pt

.PHONY: venv install install-dev check-python \
        prepare train export evaluate evaluate-weather \
        sweep benchmark bandwidth test lint

check-python:
	@$(PYTHON) --version

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

install-dev: venv
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements-dev.txt

# --- ML pipeline ---

prepare:
	$(ACTIVATE) && python ml/prepare_dataset.py --pklot-dir $(PKLOT_DIR)

train:
	$(ACTIVATE) && python ml/train.py --variant $(VARIANT) --device $(DEVICE)

export:
	$(ACTIVATE) && python ml/export.py --weights $(WEIGHTS)

evaluate:
	$(ACTIVATE) && python ml/evaluate.py --weights artifacts/models/best.pt --full

evaluate-weather:
	$(ACTIVATE) && python ml/evaluate.py --weights artifacts/models/best.pt --per-weather

sweep:
	$(ACTIVATE) && python ml/evaluate.py --weights artifacts/models/best.pt --sweep

benchmark:
	$(ACTIVATE) && python edge/benchmark.py --model artifacts/models/best.pt

bandwidth:
	$(ACTIVATE) && python ml/bandwidth.py

# --- Quality ---

test:
	$(ACTIVATE) && pytest

lint:
	$(ACTIVATE) && ruff check . && black --check .
