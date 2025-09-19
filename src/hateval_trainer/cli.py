from __future__ import annotations
import argparse
from pathlib import Path
import io
from contextlib import redirect_stdout, redirect_stderr

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
    _vlog("TensorFlow configured", verbose)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hateval-train",
        description="HatEval ES trainer (classic GRU as feature extractor + hybrid VQC head)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=Path("HatEvalES.csv"), help="Path to HatEval ES CSV")
    p.add_argument("--output", type=Path, default=Path("model_classic.keras"), help="Output .keras path (classic backbone)")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs for the classic backbone")
    # Siempre híbrido: mantenemos solo el control de epochs del híbrido
    p.add_argument("--hybrid-epochs", type=int, default=None, help="Epochs for the hybrid head (defaults to --epochs)")
    p.add_argument("--vocab", type=int, default=1000, help="Vocabulary size for TextVectorization")
    p.add_argument("--seq-len", type=int, default=120, help="Max sequence length for TextVectorization")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Preferred device")
    p.add_argument("--no-plots", action="store_true", help="Disable plotting (useful on servers)")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    p.add_argument("--show-classic-logs", action="store_true", help="Show epochs/metrics of the classic model (hidden by default)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    # Config (forzamos always-hybrid; el clásico es extractor)
    cfg = Config(
        csv_path=args.csv,
        output_model=args.output,
        epochs=args.epochs,
        hybrid_epochs=args.hybrid_epochs or args.epochs,
        use_hybrid=True,                 # <- siempre híbrido
        vocab_size=args.vocab,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        no_plots=args.no_plots,
        verbose=args.verbose,
    )

    # Configurar TF
    _configure_tf(args.device, cfg.verbose)

    # 1) Carga, limpieza y balanceo (global, como tu flujo original)
    df = load_dataset(cfg.csv_path)
    df = clean_text(df, verbose=cfg.verbose)
    df = balance_classes(df)

    # 2) Datasets y vectorizador
    X_test, y_test, train_ds, test_ds, vectorizer, num_classes = build_datasets(df, cfg)
    _log(f"Detected classes: {num_classes}")

    # 3) Modelo clásico (backbone) — ENTRENAR EN SILENCIO salvo --show-classic-logs
    classic = build_gru_model(vectorizer, cfg.vocab_size, cfg.embedding_dim, num_classes)

    if args.show_classic_logs:
        _log("Training classic backbone (showing logs)...")
        # Mostramos todo
        _classic_capture = None
        train_and_evaluate(classic, X_test, y_test, train_ds, test_ds, cfg)
    else:
        _log("Training classic backbone...")
        # Capturamos toda la salida (epochs + report) para no imprimirla ahora
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            train_and_evaluate(classic, X_test, y_test, train_ds, test_ds, cfg)
        _classic_capture = buf.getvalue()

    classic.save(cfg.output_model)
    _log(f"Saved classic model to: {cfg.output_model}")

    # 4) Híbrido (este sí imprime epochs y métricas)
    _log("Building hybrid quantum model...")
    hybrid = build_hybrid_model(classic, num_classes)

    # Respetar --hybrid-epochs si se pasa
    epochs_backup = cfg.epochs
    if cfg.hybrid_epochs and cfg.hybrid_epochs != cfg.epochs:
        cfg.epochs = cfg.hybrid_epochs

    _log("Training hybrid model...")
    # Aquí sí queremos ver progreso y reporte
    # Además, capturamos el reporte para mostrar un resumen final limpio
    buf_h = io.StringIO()
    # Dejamos que las epochs salgan a consola; capturamos SOLO el texto del final usando train_and_evaluate que imprime al terminar
    # Para asegurar captura del informe, redirigimos todo; luego reimprimimos el buffer para conservar el comportamiento visible.
    # Alternativa: dejamos sin captura y luego no lo duplicamos en el resumen.
    # Optamos por NO duplicar: dejamos visible aquí y en el resumen solo lo reetiquetamos.
    train_and_evaluate(hybrid, X_test, y_test, train_ds, test_ds, cfg)

    # Restaurar epochs del clásico si los cambiamos
    cfg.epochs = epochs_backup

if __name__ == "__main__":
    main()
