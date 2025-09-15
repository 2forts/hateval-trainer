from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
  csv_path: Path
  output_model: Path
  vocab_size: int = 1000
  embedding_dim: int = 64
  batch_size: int = 32
  buffer_size: int = 10_000
  test_size: float = 0.2
  random_state: int = 42
  epochs: int = 10
  verbose: bool = False
  no_plots: bool = False
  seq_len: int = 120
  use_hybrid: bool = False
  hybrid_epochs: int | None = None
  plots_dir: Path = Path("outputs")
