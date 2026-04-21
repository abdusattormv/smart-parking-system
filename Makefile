PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

DEVICE ?= mps
PKLOT_DIR ?= datasets/pklot_raw
CNRPARK_DIR ?=
STAGE1_VARIANT ?= s
STAGE2_VARIANT ?= n
STAGE1_WEIGHTS ?= runs/stage1_det/yolov8$(STAGE1_VARIANT)_stage1/weights/best.pt
STAGE2_WEIGHTS ?= runs/stage2_cls/yolov8$(STAGE2_VARIANT)_stage2/weights/best.pt
BENCHMARK_IMAGE ?= samples/demo.jpg
BENCHMARK_ROI ?= 50 100 200 250

.PHONY: venv install install-dev check-python \
	prepare-stage1 prepare-stage2 prepare-single-model \
	train-stage1 train-stage2 train-stage2-all \
	evaluate-stage1 evaluate-stage2 compare-stage2 sweep-stage2 \
	export-stage2 benchmark-stage2 bandwidth stability test lint finalize

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

prepare-stage1:
	$(ACTIVATE) && python ml/prepare_dataset.py --stage1 --pklot-dir $(PKLOT_DIR)

prepare-stage2:
	$(ACTIVATE) && python ml/prepare_dataset.py --stage2 --pklot-dir $(PKLOT_DIR) $(if $(CNRPARK_DIR),--cnrpark-dir $(CNRPARK_DIR),)

prepare-single-model:
	$(ACTIVATE) && python ml/prepare_dataset.py --single-model --pklot-dir $(PKLOT_DIR)

train-stage1:
	$(ACTIVATE) && python ml/train.py --stage1 --variant $(STAGE1_VARIANT) --device $(DEVICE)

train-stage2:
	$(ACTIVATE) && python ml/train.py --stage2 --variant $(STAGE2_VARIANT) --device $(DEVICE)

train-stage2-all:
	$(ACTIVATE) && python ml/train.py --stage2 --variant n --device $(DEVICE)
	$(ACTIVATE) && python ml/train.py --stage2 --variant s --device $(DEVICE)
	$(ACTIVATE) && python ml/train.py --stage2 --variant m --device $(DEVICE)

evaluate-stage1:
	$(ACTIVATE) && python ml/evaluate.py --stage1 --weights $(STAGE1_WEIGHTS) --split val --device $(DEVICE)

evaluate-stage2:
	$(ACTIVATE) && python ml/evaluate.py --stage2 --weights $(STAGE2_WEIGHTS) --split val --device $(DEVICE)

compare-stage2:
	$(ACTIVATE) && python ml/evaluate.py --stage2 --split val --device $(DEVICE) --compare \
		runs/stage2_cls/yolov8n_stage2/weights/best.pt \
		runs/stage2_cls/yolov8s_stage2/weights/best.pt \
		runs/stage2_cls/yolov8m_stage2/weights/best.pt

sweep-stage2:
	$(ACTIVATE) && python ml/evaluate.py --stage2 --weights $(STAGE2_WEIGHTS) --split val --device $(DEVICE) --sweep

export-stage2:
	$(ACTIVATE) && python ml/export.py --weights $(STAGE2_WEIGHTS) --imgsz 64

benchmark-stage2:
	$(ACTIVATE) && python edge/benchmark.py \
		--task classify \
		--image $(BENCHMARK_IMAGE) \
		--model $(STAGE2_WEIGHTS) \
		--imgsz 64 \
		--roi $(BENCHMARK_ROI)

bandwidth:
	$(ACTIVATE) && python ml/bandwidth.py

stability:
	$(ACTIVATE) && python edge/stability_test.py \
		--image $(BENCHMARK_IMAGE) \
		--stage1-detector \
		--stage1-model $(STAGE1_WEIGHTS) \
		--stage2-model $(STAGE2_WEIGHTS) \
		--duration 1800

finalize:
	$(ACTIVATE) && python ml/finalize.py

test:
	$(ACTIVATE) && pytest -q

lint:
	$(ACTIVATE) && ruff check . && black --check .
