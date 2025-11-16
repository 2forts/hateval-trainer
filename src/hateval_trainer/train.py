from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import Config

def train_and_evaluate(model, X_test, y_test, train_ds, test_ds, cfg, fit_verbose=1, print_report=True):
    """Train (with configurable verbosity) and evaluate, returning metrics and the text report."""
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    # Standard callbacks (early stopping and LR scheduling)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    # Supervised training with verbosity control
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=cfg.epochs,
        callbacks=cbs,
        verbose=fit_verbose,
    )

    # Evaluation on the provided test split
    y_logits = model.predict(tf.convert_to_tensor(X_test), batch_size=cfg.batch_size, verbose=0)
    y_pred = np.argmax(y_logits, axis=1)

    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    if print_report:
        print(report)

    # Return everything the caller may want to assemble a final summary
    return {
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_logits": y_logits,
    }

def train_hybrid_eager(forward, trainables, X_test, y_test, train_ds, test_ds, cfg: Config):
    """Train the hybrid model in eager mode with a custom loop (minimal example)."""
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
        # One full pass over the training dataset
        total_loss = 0.0
        batches = 0
        for xb, yb in train_ds:
            loss = train_step(xb, yb)
            total_loss += loss
            batches += 1
        avg_loss = float(total_loss / max(batches, 1))
        print(f"[HYBRID] Epoch {epoch+1}/{cfg.epochs} - loss: {avg_loss:.4f}")

    # Evaluation on the test dataset
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


# === Stratified k-fold CV with per-fold reports (optionally saved) ===
def cross_validate(df, cfg, k=5, label_names=None, model_builder_factory=None,
                   balance_train: bool = False):
    """
    Stratified k-fold CV with three modes:

      - Classical mode (default): train and evaluate the GRU backbone only.
      - Hybrid mode (cfg.cv_use_hybrid=True): TWO-PHASE TRAINING PER FOLD
          Phase 1) train the classical backbone end-to-end.
          Phase 2) build the hybrid model (backbone is frozen) and train only the quantum+dense head.
      - Dense ablation mode (cfg.cv_use_dense_ablation=True):
          Same two-phase protocol as the hybrid, but replacing the VQC with a small
          classical dense head (~75 parameters) to act as an ablation of the quantum layer.

    Also includes:
      - Per-fold class weights (scalable with cfg.cw_scale, default 1.0) — this is NOT resampling.
      - Optional per-fold decision-threshold tuning for binary tasks (cfg.tune_threshold).
      - Optional saving of per-fold reports and confusion matrices if cfg.cv_save_dir is set.
      - Standard callbacks: EarlyStopping + ReduceLROnPlateau.
    """
    import numpy as np
    import tensorflow as tf
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
    from sklearn.utils import resample
    from sklearn.utils.class_weight import compute_class_weight

    # Global seed for reproducibility (best-effort)
    try:
        tf.keras.utils.set_random_seed(getattr(cfg, "random_state", 42))
    except Exception:
        pass

    # Options
    save_dir = getattr(cfg, "cv_save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    cw_scale = float(getattr(cfg, "cw_scale", 1.0))
    tune_threshold = bool(getattr(cfg, "tune_threshold", False))
    use_hybrid = bool(getattr(cfg, "cv_use_hybrid", False))
    use_dense_ablation = bool(getattr(cfg, "cv_use_dense_ablation", False))

    # Si por accidente se activan ambos, damos prioridad al híbrido y avisamos
    if use_hybrid and use_dense_ablation:
        print("[WARN] Both cv_use_hybrid and cv_use_dense_ablation are True; "
              "giving priority to the hybrid (quantum) head.", flush=True)
        use_dense_ablation = False

    X = df["text"].values
    y = df["target"].values

    skf = StratifiedKFold(
        n_splits=k, shuffle=True,
        random_state=getattr(cfg, "random_state", 42),
    )

    rows, accs, f1s = [], [], []
    fold = 1

    for tr, va in skf.split(X, y):
        # Per-fold seed (optional but useful for stability)
        try:
            tf.keras.utils.set_random_seed(getattr(cfg, "random_state", 42) + fold)
        except Exception:
            pass

        X_train, X_val = X[tr], X[va]
        y_train, y_val = y[tr], y[va]

        # (Optional) upsampling ONLY on the training portion of the fold
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

        # Vectorizer adapted ONLY on the training portion of the fold
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=getattr(cfg, "vocab_size", 20000),
            output_sequence_length=getattr(cfg, "seq_len", 128),
            output_mode="int",
        )
        ds_xtr = tf.data.Dataset.from_tensor_slices(X_train).batch(getattr(cfg, "batch_size", 32))
        vectorizer.adapt(ds_xtr)

        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    .shuffle(getattr(cfg, "buffer_size", 10000),
                             seed=getattr(cfg, "random_state", 42)+fold)
                    .batch(getattr(cfg, "batch_size", 32))
                    .prefetch(tf.data.AUTOTUNE))
        val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                  .batch(getattr(cfg, "batch_size", 32))
                  .prefetch(tf.data.AUTOTUNE))

        num_classes = int(np.unique(y).size)

        # === Per-fold class weights (this is NOT resampling; it respects "no balancing") ===
        classes = np.unique(y_train)
        base_cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        # Scale the most underrepresented class (highest weight) if desired
        idx_max = int(np.argmax(base_cw))
        scaled_cw = base_cw.copy()
        scaled_cw[idx_max] = scaled_cw[idx_max] * cw_scale
        class_weights = {int(c): float(w) for c, w in zip(classes, scaled_cw)}
        # ================================================================================

        # ===== Build and train backbone and (optionally) hybrid/ablation head in two phases =====
        if model_builder_factory is None:
            from .models import build_gru_model, build_hybrid_model, build_dense_ablation_model

            # Phase 1: classical backbone
            classic = build_gru_model(
                vectorizer=vectorizer,
                vocab_size=getattr(cfg, "vocab_size", 20000),
                embedding_dim=getattr(cfg, "embedding_dim", 128),
                num_classes=num_classes,
            )

            # Shared callbacks
            cbs = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            ]

            if use_hybrid or use_dense_ablation:
                # --- Train backbone (phase 1) ---
                if getattr(cfg, "verbose", False):
                    mode = "hybrid" if use_hybrid else "dense_ablation"
                    print(f"[CV fold {fold}/{k}] Phase 1: training classical backbone for {mode} ...",
                          flush=True)
                classic.fit(
                    train_ds,
                    epochs=getattr(cfg, "epochs", 5),
                    validation_data=val_ds,
                    callbacks=cbs,
                    verbose=1 if getattr(cfg, "verbose", False) else 0,
                    class_weight=class_weights,
                )

                # --- Build the second-stage head ---
                if use_hybrid:
                    # Hybrid quantum head
                    head_model = build_hybrid_model(classic, num_classes)
                else:
                    # Classical dense ablation head
                    head_model = build_dense_ablation_model(classic, num_classes)

                # --- Train second-stage head (phase 2) ---
                epochs_backup = getattr(cfg, "epochs", 5)
                hy_epochs = getattr(cfg, "hybrid_epochs", epochs_backup)
                if getattr(cfg, "verbose", False):
                    mode = "hybrid" if use_hybrid else "dense_ablation"
                    print(f"[CV fold {fold}/{k}] Phase 2: training {mode} head ... (epochs={hy_epochs})",
                          flush=True)
                head_model.fit(
                    train_ds,
                    epochs=hy_epochs,
                    validation_data=val_ds,
                    callbacks=cbs,
                    verbose=1 if getattr(cfg, "verbose", False) else 0,
                    class_weight=class_weights,
                )
                model = head_model  # evaluate with the hybrid or ablation head
            else:
                # Backbone-only (classical mode)
                if getattr(cfg, "verbose", False):
                    print(f"[CV fold {fold}/{k}] Training classical backbone (no hybrid, no ablation) ...",
                          flush=True)
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
            # If an external factory is provided, we assume it encapsulates the two-phase logic if needed
            model = model_builder_factory(vectorizer, num_classes)()

        # ===========================================================================

        # --- Inference on validation ---
        y_logits = []
        for xb, _ in val_ds:
            y_logits.append(model(xb, training=False))
        y_logits = tf.concat(y_logits, axis=0).numpy()

        # Decision rule: optional threshold tuning for binary tasks
        if num_classes == 2:
            # Numerically stable softmax → probability of class 1 (HS)
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

        # Per-fold metrics
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        accs.append(acc); f1s.append(macro_f1)

        rep = classification_report(
            y_val, y_pred, digits=3,
            target_names=label_names if label_names is not None else None
        )

        if getattr(cfg, "verbose", False):
            mode = "hybrid" if use_hybrid else ("dense_ablation" if use_dense_ablation else "classic")
            print(f"\n[CV fold {fold}/{k}] (mode={mode}, cw_scale={cw_scale}, tune_thr={tune_threshold})",
                  flush=True)
            print(rep)
            print(f"Accuracy={acc:.4f} Macro-F1={macro_f1:.4f}")

        if save_dir is not None:
            (save_dir / f"fold_{fold:02d}_report.txt").write_text(rep)
            cm = confusion_matrix(y_val, y_pred)
            fig = _cm_figure(cm, label_names)
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
    """Return a matplotlib Figure with the confusion matrix annotated."""
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
