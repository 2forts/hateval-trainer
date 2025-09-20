from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import Config


def train_and_evaluate(model, X_test, y_test, train_ds, test_ds, cfg, fit_verbose=1, print_report=True):
    """Entrena (con verbosidad configurable) y evalúa, devolviendo métricas e informe."""
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    # callbacks como tuvieses ya (early stopping, reduce LR, etc.)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    # entrenamiento con control de verbosidad
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=cfg.epochs,
        callbacks=cbs,
        verbose=fit_verbose,
    )

    # evaluación
    y_logits = model.predict(tf.convert_to_tensor(X_test), batch_size=cfg.batch_size, verbose=0)
    y_pred = np.argmax(y_logits, axis=1)

    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    if print_report:
        print(report)

    # Devuelve por si el caller quiere componer un resumen final
    return {
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_logits": y_logits,
    }


def train_hybrid_eager(forward, trainables, X_test, y_test, train_ds, test_ds, cfg: Config):
    """Train the hybrid model in eager mode with a custom loop."""
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import classification_report

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function(experimental_relax_shapes=True)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = forward(xb)
            loss = loss_fn(yb, logits)
        grads = tape.gradient(loss, trainables)
        optimizer.apply_gradients(zip(grads, trainables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def predict_step(xb):
        return forward(xb)

    for epoch in range(cfg.epochs):
        # Train
        total_loss = 0.0
        batches = 0
        for xb, yb in train_ds:
            loss = train_step(xb, yb)
            total_loss += loss
            batches += 1
        avg_loss = float(total_loss / max(batches, 1))
        print(f"[HYBRID] Epoch {epoch+1}/{cfg.epochs} - loss: {avg_loss:.4f}")

    # Evaluate
    y_logits = []
    y_true = []
    for xb, yb in test_ds:
        y_logits.append(predict_step(xb))
        y_true.append(yb)
    y_logits = tf.concat(y_logits, axis=0).numpy()
    y_true = tf.concat(y_true, axis=0).numpy()
    y_pred = np.argmax(y_logits, axis=1)

    print("\n[HYBRID] Classification report:\n")
    print(classification_report(y_true, y_pred, digits=3))


# === Stratified k-fold CV con informes por pliegue (opcionalmente guardados) ===

def cross_validate(df, cfg, k=5, label_names=None, model_builder_factory=None,
                   balance_train: bool = False):
    """
    Stratified k-fold CV con dos modos:
      - Modo clásico (por defecto): solo backbone GRU.
      - Modo híbrido (cfg.cv_use_hybrid=True): ENTRENAMIENTO EN DOS FASES POR PLIEGUE:
          Fase 1) entrenar backbone clásico end-to-end.
          Fase 2) construir híbrido (backbone congelado) y entrenar solo la cabeza cuántica+densa.
    También incluye:
      - Class weights por pliegue (escalables con cfg.cw_scale, default=1.0).
      - Umbral de decisión opcional por pliegue (binario) con cfg.tune_threshold.
      - Guardado opcional de reports y matrices de confusión si cfg.cv_save_dir.
      - Callbacks: EarlyStopping + ReduceLROnPlateau.
    """
    import numpy as np
    import tensorflow as tf
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
    from sklearn.utils import resample
    from sklearn.utils.class_weight import compute_class_weight

    # Semillas
    try:
        tf.keras.utils.set_random_seed(getattr(cfg, "random_state", 42))
    except Exception:
        pass

    # Opciones
    save_dir = getattr(cfg, "cv_save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    cw_scale = float(getattr(cfg, "cw_scale", 1.0))
    tune_threshold = bool(getattr(cfg, "tune_threshold", False))
    use_hybrid = bool(getattr(cfg, "cv_use_hybrid", False))

    X = df["text"].values
    y = df["target"].values

    skf = StratifiedKFold(
        n_splits=k, shuffle=True,
        random_state=getattr(cfg, "random_state", 42),
    )

    rows, accs, f1s = [], [], []
    fold = 1

    for tr, va in skf.split(X, y):
        # Semilla por fold
        try:
            tf.keras.utils.set_random_seed(getattr(cfg, "random_state", 42) + fold)
        except Exception:
            pass

        X_train, X_val = X[tr], X[va]
        y_train, y_val = y[tr], y[va]

        # (Opcional) upsampling SOLO en train del fold
        if balance_train:
            classes_bt, counts = np.unique(y_train, return_counts=True)
            n_max = counts.max()
            Xb, yb = [], []
            for c in classes_bt:
                Xi = X_train[y_train == c]; yi = y_train[y_train == c]
                Xi_res, yi_res = resample(
                    Xi, yi, replace=True, n_samples=n_max,
                    random_state=getattr(cfg, "random_state", 42) + fold
                )
                Xb.append(Xi_res); yb.append(yi_res)
            X_train = np.concatenate(Xb); y_train = np.concatenate(yb)

        # Vectorizer SOLO con train del fold
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=getattr(cfg, "vocab_size", 20000),
            output_sequence_length=getattr(cfg, "seq_len", 128),
            output_mode="int",
        )
        ds_xtr = tf.data.Dataset.from_tensor_slices(X_train).batch(getattr(cfg, "batch_size", 32))
        vectorizer.adapt(ds_xtr)

        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    .shuffle(getattr(cfg, "buffer_size", 10000), seed=getattr(cfg, "random_state", 42)+fold)
                    .batch(getattr(cfg, "batch_size", 32))
                    .prefetch(tf.data.AUTOTUNE))
        val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                  .batch(getattr(cfg, "batch_size", 32))
                  .prefetch(tf.data.AUTOTUNE))

        num_classes = int(np.unique(y).size)

        # === Pesos de clase por pliegue (NO es resampling; cumple "sin balancing") ===
        classes = np.unique(y_train)
        base_cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        # Escalamos la clase con mayor peso (minoritaria) para ajustar precisión/recobrado si se desea
        idx_max = int(np.argmax(base_cw))
        scaled_cw = base_cw.copy()
        scaled_cw[idx_max] = scaled_cw[idx_max] * cw_scale
        class_weights = {int(c): float(w) for c, w in zip(classes, scaled_cw)}
        # ============================================================================

        # ===== Builder del backbone e (opcional) híbrido en dos fases =====
        if model_builder_factory is None:
            from .models import build_gru_model, build_hybrid_model
            # Fase 1: backbone clásico
            classic = build_gru_model(
                vectorizer=vectorizer,
                vocab_size=getattr(cfg, "vocab_size", 20000),
                embedding_dim=getattr(cfg, "embedding_dim", 128),
                num_classes=num_classes,
            )

            # Callbacks compartidos
            cbs = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            ]

            if use_hybrid:
                # --- ENTRENAR BACKBONE (fase 1) ---
                if getattr(cfg, "verbose", False):
                    print(f"[CV fold {fold}/{k}] Phase 1: training classical backbone ...")
                classic.fit(
                    train_ds,
                    epochs=getattr(cfg, "epochs", 5),
                    validation_data=val_ds,
                    callbacks=cbs,
                    verbose=1 if getattr(cfg, "verbose", False) else 0,
                    class_weight=class_weights,
                )

                # Construir híbrido congelando el extractor (build_hybrid_model debe encargarse de ello)
                hybrid = build_hybrid_model(classic, num_classes)

                # --- ENTRENAR CABEZA HÍBRIDA (fase 2) ---
                # Respetamos hybrid_epochs si está definido
                epochs_backup = getattr(cfg, "epochs", 5)
                hy_epochs = getattr(cfg, "hybrid_epochs", epochs_backup)
                if getattr(cfg, "verbose", False):
                    print(f"[CV fold {fold}/{k}] Phase 2: training hybrid head ... (epochs={hy_epochs})")
                hybrid.fit(
                    train_ds,
                    epochs=hy_epochs,
                    validation_data=val_ds,
                    callbacks=cbs,
                    verbose=1 if getattr(cfg, "verbose", False) else 0,
                    class_weight=class_weights,
                )
                model = hybrid  # evaluar con el híbrido
            else:
                # Solo backbone (modo clásico)
                if getattr(cfg, "verbose", False):
                    print(f"[CV fold {fold}/{k}] Training classical backbone (no hybrid) ...")
                classic.fit(
                    train_ds,
                    epochs=getattr(cfg, "epochs", 5),
                    validation_data=val_ds,
                    callbacks=cbs,
                    verbose=1 if getattr(cfg, "verbose", False) else 0,
                    class_weight=class_weights,
                )
                model = classic
        else:
            # Si te pasan un factory externo, lo usamos tal cual (se asume que implementa la lógica de dos fases si procede)
            model = model_builder_factory(vectorizer, num_classes)()
        # =================================================================

        # --- Predicción en validación ---
        y_logits = []
        for xb, _ in val_ds:
            y_logits.append(model(xb, training=False))
        y_logits = tf.concat(y_logits, axis=0).numpy()

        # Decisión: umbral optativo para binario
        if num_classes == 2:
            # Softmax estable -> prob de clase 1 (HS)
            z = y_logits - y_logits.max(axis=1, keepdims=True)
            expz = np.exp(z)
            p1 = expz[:, 1] / expz.sum(axis=1, keepdims=True)[:, 0]
            if tune_threshold:
                ts = np.linspace(0.3, 0.7, 81)
                best_t, best_f1 = 0.5, -1.0
                for t in ts:
                    y_pred_t = (p1 >= t).astype(int)
                    f1 = f1_score(y_val, y_pred_t, average="macro")
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                y_pred = (p1 >= best_t).astype(int)
            else:
                y_pred = (p1 >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_logits, axis=1)

        # Métricas del fold
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        accs.append(acc); f1s.append(macro_f1)

        rep = classification_report(
            y_val, y_pred, digits=3,
            target_names=label_names if label_names is not None else None
        )

        if getattr(cfg, "verbose", False):
            print(f"\n[CV fold {fold}/{k}] (hybrid={use_hybrid}, cw_scale={cw_scale}, tune_thr={tune_threshold})")
            print(rep)
            print(f"Accuracy={acc:.4f} Macro-F1={macro_f1:.4f}")

        if save_dir is not None:
            (save_dir / f"fold_{fold:02d}_report.txt").write_text(rep)
            cm = confusion_matrix(y_val, y_pred)
            fig = _cm_figure(cm, label_names)  # asegúrate de tener esta utilidad en train.py
            fig.savefig(save_dir / f"fold_{fold:02d}_cm.png", dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt; plt.close(fig)

        rows.append({"fold": fold, "accuracy": acc, "macro_f1": macro_f1})
        fold += 1

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_f1,  std_f1  = float(np.mean(f1s)),  float(np.std(f1s))

    print("\nCross-validation summary:")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    if save_dir is not None and rows:
        pd.DataFrame(rows + [{
            "fold": "mean±std",
            "accuracy": f"{mean_acc:.4f}±{std_acc:.4f}",
            "macro_f1": f"{mean_f1:.4f}±{std_f1:.4f}",
        }]).to_csv(save_dir / "summary.csv", index=False)

    return {
        "cv_accuracy_mean": mean_acc,
        "cv_accuracy_std": std_acc,
        "cv_macro_f1_mean": mean_f1,
        "cv_macro_f1_std": std_f1,
    }

def _cm_figure(cm: np.ndarray, labels: list | None):
    """Devuelve una figura de matplotlib con la matriz de confusión."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    n = cm.shape[0]
    tick = labels if labels is not None else [str(i) for i in range(n)]
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=tick, yticklabels=tick,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
