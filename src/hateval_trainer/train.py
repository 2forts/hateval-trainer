from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .config import Config

def train_and_evaluate(model, X_test, y_test, train_ds, test_ds, cfg: Config):
  """Fit the model and print evaluation metrics. Optionally save confusion matrix plot."""
  import tensorflow as tf
  
  callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
  ]
   
  model.fit(train_ds, validation_data=test_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=1)
    
  # Predict on raw strings (the model's input is string)
  y_logits = model.predict(tf.convert_to_tensor(X_test), batch_size=cfg.batch_size, verbose=0)
  y_pred = np.argmax(y_logits, axis=1)
    
  print("\nClassification report:\n")
  print(classification_report(y_test, y_pred, digits=3))
    
  # Optional confusion matrix figure
  if not cfg.no_plots:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    cm = confusion_matrix(y_test, y_pred)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cfg.plots_dir / "cm_classic.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – Classic")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved confusion matrix to: {fig_path}")
