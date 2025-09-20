#!/usr/bin/env bash
set -euo pipefail

# === Example training commands for hateval-trainer ===
# Adjust --device to "cpu" or "gpu" depending on your setup.

# ----------------------------------------------------
# 1) HatEval (official split, hybrid model)
# ----------------------------------------------------
PYTHONPATH=src python -m hateval_trainer.cli \
  --csv HatEvalES.csv \
  --output modelo_clasico_base.keras \
  --epochs 20 \
  --hybrid-epochs 15 \
  --vocab 10000 \
  --batch-size 32 \
  --seq-len 120 \
  --device cpu \
  --verbose --no-plots \
  --no-balance \
  --show-classic-logs

# ----------------------------------------------------
# 2) HaterNet (10-fold cross-validation, hybrid model)
# ----------------------------------------------------
PYTHONPATH=src python -m hateval_trainer.cli \
  --csv HaterNet.csv \
  --epochs 20 \
  --hybrid-epochs 15 \
  --vocab 10000 \
  --batch-size 32 \
  --seq-len 120 \
  --device cpu \
  --verbose --no-plots \
  --no-balance \
  --crossval 10 \
  --cv-save --cv-dir outputs/cv_hybrid \
  --cv-use-hybrid
