# HatEval Trainer

A clean, reproducible training pipeline for Spanish datasets, featuring a classic Bidirectional GRU baseline and an hybrid quantum head powered by PennyLane.

hateval-trainer/
├─ LICENSE
├─ README.md
├─ .gitignore
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
├─ .pre-commit-config.yaml
├─ scripts/
│   ├─ download_nltk.py
│   └─ examples.sh
├─ src/
│   └─ hateval_trainer/
│   ├─ __init__.py
│   ├─ config.py
│   ├─ data.py
│   ├─ models.py
│   ├─ train.py
│   └─ cli.py
└─ tests/
    └─ test_smoke.py

## Features
- Text preprocessing: lowercasing, punctuation stripping, Spanish stopwords, basic lemmatization.
- Class balancing via upsampling.
- `TextVectorization` + stacked BiGRU baseline.
- Hybrid quantum-classical head (VQC) with PennyLane.
- Confusion matrix plot (optional) and classification report.
- CLI (`hateval-train`) and modular source code (`src/hateval_trainer`).

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# optional dev tooling
pip install pre-commit black ruff pytest && pre-commit install
