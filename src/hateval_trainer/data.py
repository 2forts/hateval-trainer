# src/hateval_trainer/data.py
from __future__ import annotations
import re
import string
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# NLTK is relatively lightweight and improves UX for Spanish stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .config import Config


# ---------- Loading & preprocessing ----------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load HatEval ES CSV (semicolon-separated). Expect columns text/HS.

    If present, drop auxiliary columns like id/TR/AG and rename to (mensaje, intensidad).
    """
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    df = df.rename(columns={"text": "mensaje", "HS": "intensidad"})
    for col in ("id", "TR", "AG"):
        if col in df.columns:
            df = df.drop(columns=[col])
    if not {"mensaje", "intensidad"}.issubset(df.columns):
        raise ValueError("CSV must contain 'mensaje'/'intensidad' columns (or 'text'/'HS').")
    return df


def _ensure_nltk(verbose: bool = False) -> tuple[set[str], WordNetLemmatizer]:
    """Ensure NLTK resources are installed and return Spanish stopwords + lemmatizer."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=not verbose)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=not verbose)
    sw = set(stopwords.words("spanish"))
    lemmatizer = WordNetLemmatizer()
    return sw, lemmatizer


def clean_text(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Normalize text: lowercase, strip punctuation, remove Spanish stopwords, lemmatize.

    Note: WordNet lemmatizer is English-focused; for better Spanish lemmatization,
    consider spaCy (es_core_news_sm) as a drop-in replacement.
    """
    stop_words, lemmatizer = _ensure_nltk(verbose)

    def strip_punct(t: str) -> str:
        return re.sub(f"[{re.escape(string.punctuation)}]", "", t)

    def norm(t: str) -> str:
        return " ".join(
            lemmatizer.lemmatize(w) for w in t.split() if w and w not in stop_words
        )

    out = df.copy()
    out["mensaje"] = out["mensaje"].astype(str).str.lower().map(strip_punct).map(norm)
    return out


def balance_classes(df: pd.DataFrame, label: str = "intensidad", seed: int = 42) -> pd.DataFrame:
    """Upsample minority classes to the majority class size using replacement, then shuffle."""
    parts = []
    max_count = df[label].value_counts().max()
    for lab in df[label].unique():
        parts.append(resample(df[df[label] == lab], replace=True, n_samples=max_count, random_state=seed))
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_datasets(df: pd.DataFrame, cfg: Config):
    """Return (X_test, y_test, train_ds, test_ds, vectorizer, num_classes)."""
    import tensorflow as tf

    X = df["mensaje"].astype(str).values
    y = df["intensidad"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(cfg.buffer_size)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(cfg.batch_size).prefetch(
        tf.data.AUTOTUNE
    )

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=cfg.vocab_size, output_sequence_length=cfg.seq_len
    )
    vectorizer.adapt(train_ds.map(lambda x, y: x))

    num_classes = int(pd.Series(y).nunique())
    return X_test, y_test, train_ds, test_ds, vectorizer, num_classes
