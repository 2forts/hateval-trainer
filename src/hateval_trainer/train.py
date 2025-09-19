from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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
    model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs, callbacks=cbs, verbose=fit_verbose)

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
