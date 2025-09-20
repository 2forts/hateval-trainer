# HatEval Trainer

A clean, reproducible training pipeline for Spanish datasets, featuring a classic Bidirectional GRU baseline and an hybrid quantum head powered by PennyLane.

## Repository Structure

```text
hateval-trainer/
├── LICENSE
├── README.md
├── .gitignore
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # Runtime dependencies
├── Makefile                # Handy shortcuts (train, lint, test…)
│
├── scripts/                # Utility scripts
│   ├── download_nltk.py    # Pre-download NLTK resources
│   └── examples.sh         # Example training commands
│
├── src/
│   └── hateval_trainer/    # Main package
│       ├── __init__.py
│       ├── config.py       # Config dataclass
│       ├── data.py         # Data loading & preprocessing
│       ├── models.py       # GRU & hybrid quantum models
│       ├── train.py        # Training & evaluation loop
│       └── cli.py          # Command-line interface (hateval-train)
│
└── tests/                  # Pytest-based unit tests
    └── test_smoke.py       # Minimal smoke test
```

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
pip install black ruff pytest
```

## Usage

After installation, you can run the training pipeline via the CLI:

### HatEval (official split)

We follow the official HatEval train/test split. To train the classical backbone and then the hybrid head:

```bash
PYTHONPATH=src python -m hateval_trainer.cli \\
  --csv HatEvalES.csv \\
  --output modelo_clasico_base.keras \\
  --epochs 20 \\
  --hybrid-epochs 15 \\
  --vocab 10000 \\
  --batch-size 32 \\
  --seq-len 120 \\
  --device cpu \\
  --verbose --no-plots \\
  --no-balance \\
  --show-classic-logs
```

This will:
- Preprocess and split HatEvalES.csv.
- Train the BiGRU backbone (logs optional).
- Train the hybrid quantum head for 15 epochs.
- Save the backbone to `modelo_clasico_base.keras`.

### HaterNet (10-fold cross-validation)

For HaterNet we use stratified cross-validation without class balancing, as recommended in prior work and reviewer comments:

```bash
PYTHONPATH=src python -m hateval_trainer.cli \\
  --csv HaterNet.csv \\
  --epochs 20 \\
  --hybrid-epochs 15 \\
  --vocab 10000 \\
  --batch-size 32 \\
  --seq-len 120 \\
  --device cpu \\
  --verbose --no-plots \\
  --no-balance \\
  --crossval 10 \\
  --cv-save --cv-dir outputs/cv_hybrid \\
  --cv-use-hybrid
```

This will:
- Perform 10-fold stratified CV.
- Train backbone + hybrid head in each fold.
- Save per-fold reports and confusion matrices under `outputs/cv_hybrid/`.
- Print a summary with mean ± std accuracy and macro-F1.

### Notes
- Use `--device gpu` to enable GPU if available.
- Use `--tune-threshold` to tune decision thresholds per fold (binary tasks).
- Disable balancing with `--no-balance` to avoid data leakage.
