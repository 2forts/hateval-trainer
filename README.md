# HatEval Trainer

A clean, reproducible training pipeline for Spanish datasets, featuring a classic Bidirectional GRU baseline and an hybrid quantum head powered by PennyLane.

## Repository Structure

hateval-trainer/
├── LICENSE
├── README.md
├── .gitignore
├── pyproject.toml # Project metadata & dependencies
├── requirements.txt # Runtime dependencies
├── Makefile # Handy shortcuts (train, lint, test…)
├── .pre-commit-config.yaml # Code quality hooks
│
├── scripts/ # Utility scripts
│ ├── download_nltk.py # Pre-download NLTK resources
│ └── examples.sh # Example training commands
│
├── src/
│ └── hateval_trainer/ # Main package
│ ├── init.py
│ ├── config.py # Config dataclass
│ ├── data.py # Data loading & preprocessing
│ ├── models.py # GRU & hybrid quantum models
│ ├── train.py # Training & evaluation loop
│ └── cli.py # Command-line interface (hateval-train)
│
└── tests/ # Pytest-based unit tests
└── test_smoke.py # Minimal smoke test

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
