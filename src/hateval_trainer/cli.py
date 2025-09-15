from __future__ import annotations
import argparse
from pathlib import Path

from .config import Config
from .data import load_dataset, clean_text, balance_classes, build_datasets
from .models import build_gru_model, build_hybrid_model
from .train import train_and_evaluate

def _log(msg: str):
  print(f"[INFO] {msg}")

def _vlog(msg: str, verbose: bool):
  if verbose:
    print(f"[VERBOSE] {msg}")

def _configure_tf(device: str, verbose: bool):
  import tensorflow as tf
  
  if device == "cpu":
    try:
      tf.config.set_visible_devices([], "GPU")
      _log("Using CPU (GPU disabled by --device cpu)")
    except Exception as e:
      print(e)
  else:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
          _log(f"Detected GPU(s): {len(gpus)} (memory_growth enabled)")
      except RuntimeError as e:
        print(e)
    else:
      _log("No GPU available; falling back to CPU")
  _vlog(f"TensorFlow configured", verbose)

def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="hateval-train",
    description="HatEval ES trainer (classic GRU + optional hybrid VQC head)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--csv", type=Path, default=Path("HatEvalES.csv"), help="Path to HatEval ES CSV")
  p.add_argument("--output", type=Path, default=Path("model_classic.keras"), help="Output .keras path")
  p.add_argument("--epochs", type=int, default=10, help="Training epochs for the classic model")
  p.add_argument("--hybrid", action="store_true", help="Also build and train the hybrid quantum head")
  p.add_argument("--hybrid-epochs", type=int, default=None, help="Epochs for the hybrid model (defaults to --epochs)")
  p.add_argument("--vocab", type=int, default=1000, help="Vocabulary size for TextVectorization")
  p.add_argument("--seq-len", type=int, default=120, help="Max sequence length for TextVectorization")
  p.add_argument("--batch-size", type=int, default=32, help="Batch size")
  p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Preferred device")
  p.add_argument("--no-plots", action="store_true", help="Disable plotting (useful on servers)")
  p.add_argument("--verbose", action="store_true", help="Verbose logs")
  return p

def main(argv=None):
  args = build_argparser().parse_args(argv)
  
  cfg = Config(
    csv_path=args.csv,
    output_model=args.output,
    epochs=args.epochs,
    hybrid_epochs=args.hybrid_epochs or args.epochs,
    use_hybrid=args.hybrid,
    vocab_size=args.vocab,
    seq_len=args.seq_len,
    batch_size=args.batch_size,
    no_plots=args.no_plots,
    verbose=args.verbose,
  )
  
  # Configure TF runtime early
  _configure_tf(args.device, cfg.verbose)
  
  # Load + preprocess
  df = load_dataset(cfg.csv_path)
  df = clean_text(df, verbose=cfg.verbose)
  df = balance_classes(df)
   
  # Build datasets and model(s)
  X_test, y_test, train_ds, test_ds, vectorizer, num_classes = build_datasets(df, cfg)
  _log(f"Detected classes: {num_classes}")
  
  classic = build_gru_model(vectorizer, cfg.vocab_size, cfg.embedding_dim, num_classes)
  classic.summary()
  
  # Train classic
  train_and_evaluate(classic, X_test, y_test, train_ds, test_ds, cfg)
  classic.save(cfg.output_model)
  _log(f"Saved classic model to: {cfg.output_model}")
  
  # Hybrid (optional)
  if cfg.use_hybrid:
    _log("Building hybrid quantum model...")
    hybrid = build_hybrid_model(classic, num_classes)
    if cfg.hybrid_epochs and cfg.hybrid_epochs != cfg.epochs:
      cfg_h = Config(**{**cfg.__dict__})
      cfg_h.epochs = cfg.hybrid_epochs
      train_and_evaluate(hybrid, X_test, y_test, train_ds, test_ds, cfg_h)
    else:
      train_and_evaluate(hybrid, X_test, y_test, train_ds, test_ds, cfg)
  
    out_h = Path(cfg.output_model).with_name("model_hybrid.keras")
    hybrid.save(out_h)
    _log(f"Saved hybrid model to: {out_h}")

if __name__ == "__main__":
  main()
