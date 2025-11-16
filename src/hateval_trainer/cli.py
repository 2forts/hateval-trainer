from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from .config import Config
from .data import load_dataset, clean_text, build_datasets
from .models import build_gru_model, build_hybrid_model
from .train import train_and_evaluate  # cross_validate se importa dinámicamente si se usa


def _log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def _vlog(msg: str, verbose: bool):
    if verbose:
        print(f"[VERBOSE] {msg}", flush=True)


def _configure_tf(device: str, verbose: bool):
    import os
    import tensorflow as tf

    _vlog("Importing TensorFlow and configuring device ...", verbose)
    if device == "cpu":
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.config.set_visible_devices([], "GPU")
            _log("Using CPU (GPU disabled by --device cpu)")
        except Exception as e:
            print(e, file=sys.stderr, flush=True)
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                _log(f"Detected GPU(s): {len(gpus)} (memory_growth enabled)")
            except RuntimeError as e:
                print(e, file=sys.stderr, flush=True)
        else:
            _log("No GPU available; falling back to CPU")
    _vlog("TensorFlow configured", verbose)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hateval-train",
        description="HatEval/HaterNet ES trainer (GRU backbone + optional hybrid quantum head)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Datos / salida
    p.add_argument("--csv", type=Path, default=Path("HatEvalES.csv"), help="Path to CSV")
    p.add_argument("--output", type=Path, default=Path("model_classic.keras"), help="Output .keras path (classic backbone)")
    # Hiperparámetros
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--hybrid-epochs", type=int, default=None, help="Epochs for the hybrid head (defaults to --epochs)")
    p.add_argument("--vocab", type=int, default=1000, help="Vocabulary size for TextVectorization")
    p.add_argument("--seq-len", type=int, default=120, help="Max sequence length for TextVectorization")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Preferred device")
    # Miscelánea
    p.add_argument("--no-plots", action="store_true", help="Disable plotting (useful on servers)")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    p.add_argument("--show-classic-logs", action="store_true", help="Show epochs/metrics of the classic model (hidden by default)")
    p.add_argument("--no-balance", action="store_true", help="Disable class balancing (no upsampling in training)")
    # Cross-validation
    p.add_argument("--crossval", type=int, default=0, help="Number of folds for stratified cross-validation (0 disables)")
    p.add_argument("--cv-save", action="store_true", help="Save per-fold reports and confusion matrices during cross-validation")
    p.add_argument("--cv-dir", type=Path, default=Path("outputs/cv"), help="Directory to save cross-validation artifacts")
    p.add_argument("--cv-use-hybrid", action="store_true", help="Use the hybrid (quantum) head inside cross-validation (slower)")
    p.add_argument("--cv-use-dense-ablation", action="store_true", help="Use the classical dense head (ablation) instead of VQC inside CV")
    p.add_argument("--tune-threshold", action="store_true", help="Tune decision threshold per CV fold to maximize macro-F1 (binary only)")
    p.add_argument("--cw-scale", type=float, default=1.0, help="Scale factor for minority class weight in CV (1.0 keeps sklearn 'balanced')")
    
    return p

def main(argv=None):
    try:
        print("[BOOT] starting CLI main()", flush=True)
        args = build_argparser().parse_args(argv)
        print(f"[BOOT] args parsed: {args}", flush=True)

        # Construir Config base (respetando tu clase original)
        cfg = Config(
            csv_path=args.csv,
            output_model=args.output,
            epochs=args.epochs,
            hybrid_epochs=args.hybrid_epochs or args.epochs,
            use_hybrid=True,  # mantenemos semántica original: siempre se construye híbrido en hold-out
            vocab_size=args.vocab,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            no_plots=args.no_plots,
            verbose=args.verbose,
        )
        # Atributos dinámicos para funcionalidades nuevas, sin romper tu Config
        setattr(cfg, "no_balance", bool(args.no_balance))
        if args.cv_save:
            setattr(cfg, "cv_save_dir", args.cv_dir)
        setattr(cfg, "cv_use_hybrid", bool(args.cv_use_hybrid))
        setattr(cfg, "cv_use_dense_ablation", bool(args.cv_use_dense_ablation))
        setattr(cfg, "tune_threshold", bool(args.tune_threshold))
        setattr(cfg, "cw_scale", float(args.cw_scale))

        _vlog(f"Config built: {cfg}", cfg.verbose)

        _configure_tf(args.device, cfg.verbose)

        # Carga y limpieza (sin balanceo global -> evitamos leakage)
        _log("Loading dataset ...")
        df = load_dataset(cfg.csv_path)
        _log(f"Loaded dataframe: {len(df)} rows")

        _log("Cleaning text ...")
        df = clean_text(df, verbose=cfg.verbose)
        _vlog("Text cleaned", cfg.verbose)

        # Cross-validation (si se solicita)
        if args.crossval and args.crossval > 0:
            from .train import cross_validate  # import diferido para evitar dependencias si no se usa
            _log(f"Running stratified {args.crossval}-fold cross-validation (no balancing inside folds by default)...")
            cross_validate(df, cfg, k=args.crossval)
            _log("Cross-validation finished")
            return

        # Hold-out (split interno) — backbone + híbrido
        _log("Building datasets ...")
        X_test, y_test, train_ds, test_ds, vectorizer, num_classes = build_datasets(df, cfg)
        _log(f"Detected classes: {num_classes}")
        try:
            tb = train_ds.cardinality().numpy()
            vb = test_ds.cardinality().numpy()
            _vlog(f"Dataset cardinality: train={tb} batches, test={vb} batches", cfg.verbose)
        except Exception:
            pass

        # Backbone clásico
        _log("Building classic backbone model ...")
        classic = build_gru_model(vectorizer, cfg.vocab_size, cfg.embedding_dim, num_classes)

        if args.show_classic_logs:
            _log("Training classic backbone (showing logs) ...")
            print("[RUN] Starting Keras fit (classic) ...", flush=True)
            train_and_evaluate(classic, X_test, y_test, train_ds, test_ds, cfg)
        else:
            _log("Training classic backbone (silenced logs) ...")
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                train_and_evaluate(classic, X_test, y_test, train_ds, test_ds, cfg)
            _ = buf.getvalue()  # disponible si quisieras persistir logs

        classic.save(cfg.output_model)
        _log(f"Saved classic model to: {cfg.output_model}")

        # Híbrido (capa cuántica)
        _log("Building hybrid quantum model ...")
        hybrid = build_hybrid_model(classic, num_classes)

        # Respetar --hybrid-epochs si difiere de --epochs
        epochs_backup = cfg.epochs
        if cfg.hybrid_epochs and cfg.hybrid_epochs != cfg.epochs:
            cfg.epochs = cfg.hybrid_epochs

        _log("Training hybrid model ...")
        print("[RUN] Starting Keras fit (hybrid) ...", flush=True)
        train_and_evaluate(hybrid, X_test, y_test, train_ds, test_ds, cfg)

        # Restaurar epochs originales
        cfg.epochs = epochs_backup
        _log("Done.")

    except SystemExit as e:
        # argparse provoca SystemExit al mostrar ayuda/errores
        print(f"[EXIT] SystemExit: code={e.code}", flush=True)
        raise
    except Exception as e:
        print("[ERROR] Unhandled exception in CLI:", repr(e), file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
