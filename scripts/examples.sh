#!/usr/bin/env bash
set -euo pipefail

# CPU
hateval-train --csv HatEvalES.csv --epochs 1 --device cpu --batch-size 16 --seq-len 100 --verbose

# GPU (first GPU), with memory growth handled in code
CUDA_VISIBLE_DEVICES=0 hateval-train --csv HatEvalES.csv --epochs 1 --device gpu --batch-size 16 --seq-len 100 --verbose

# Hybrid (quantum) on CPU backend
hateval-train --csv HatEvalES.csv --epochs 1 --device cpu --batch-size 8 --seq-len 100 --verbose --hybrid
