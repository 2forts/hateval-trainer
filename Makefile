.PHONY: venv install dev lint fmt test train-cpu train-gpu nltk

PY?=python
PIP?=pip
DATA?=HatEvalES.csv
EPOCHS?=1
BATCH?=16
SEQLEN?=100

venv:
  $(PY) -m venv .venv
  . .venv/bin/activate && pip install -U pip

install:
  $(PIP) install -r requirements.txt
  $(PIP) install -e .

dev:
  $(PIP) install pre-commit black ruff pytest
  pre-commit install

lint:
  ruff check src tests

fmt:
  black src tests
  ruff check --fix src tests || true

test:
  pytest -q

nltk:
  $(PY) scripts/download_nltk.py

train-cpu:
  hateval-train --csv $(DATA) --epochs $(EPOCHS) --device cpu --batch-size $(BATCH) --seq-len $(SEQLEN) --verbose

train-gpu:
  CUDA_VISIBLE_DEVICES=0 hateval-train --csv $(DATA) --epochs $(EPOCHS) --device gpu --batch-size $(BATCH) --seq-len $(SEQLEN) --verbose
