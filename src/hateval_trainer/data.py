from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# NLTK for Spanish stopwords (lightweight and easy to install)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .config import Config


# ---------- Loading and column normalization ----------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load a CSV and normalize column names to ('text', 'target').

    Features:
      - Tries multiple encodings and separators (robust to CSV variations).
      - Accepts common English/Spanish aliases for text and label columns.
      - Normalizes labels to integer codes in the range 0..C-1.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly two columns: 'text' (str) and 'target' (int).
    """
    import pandas as pd

    # Try several (encoding, separator) combinations for robustness
    attempts = [
        ("utf-8", ";"),
        ("utf-8", ","),
        ("utf-8", None),     # infer separator (python engine)
        ("latin-1", ";"),
        ("latin-1", ","),
        ("latin-1", None),
    ]
    last_err = None
    df = None
    for enc, sep in attempts:
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=sep) if sep is not None \
                 else pd.read_csv(csv_path, encoding=enc, engine="python")
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise ValueError(f"Could not read CSV {csv_path}: {last_err}")

    # Normalize column names to lowercase, strip whitespace
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Common aliases for the text and target columns (EN + ES)
    text_aliases = {
        "text", "tweet", "sentence", "content", "tweet_text",
        "mensaje", "texto", "comentario", "contenido"
    }
    target_aliases = {
        "target", "hs", "label", "class", "y", "category",
        "intensidad", "resultado", "etiqueta", "clase", "categoria"
    }

    text_col = next((c for c in df.columns if c in text_aliases), None)
    target_col = next((c for c in df.columns if c in target_aliases), None)
    if text_col is None or target_col is None:
        raise ValueError(
            "CSV must contain text/label columns. "
            f"Found columns: {list(df.columns)}. "
            f"Expected text in one of {sorted(text_aliases)} and label in {sorted(target_aliases)}."
        )

    # Rename to canonical names
    df = df.rename(columns={text_col: "text", target_col: "target"}).copy()

    # Drop common auxiliary columns if present
    for col in ("id", "tr", "ag"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Normalize labels to consecutive integer codes (0..C-1)
    df["target"] = pd.Categorical(df["target"]).codes.astype(int)

    return df[["text", "target"]]


# ---------- Text preprocessing ----------

def _ensure_nltk(verbose: bool = False) -> tuple[set[str], WordNetLemmatizer]:
    """
    Ensure required NLTK resources are available and return
    Spanish stopwords + a lemmatizer.

    Returns
    -------
    (set[str], WordNetLemmatizer)
        Spanish stopwords and a WordNetLemmatizer instance.
    """
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
    """
    Normalize raw text with a simple, language-agnostic pipeline:
      - Lowercase
      - Remove URLs and punctuation
      - Remove Spanish stopwords
      - Lemmatize tokens with WordNet

    Notes
    -----
    WordNet lemmatization is English-centered; for best Spanish lemmatization,
    consider spaCy's 'es_core_news_sm' as a drop-in alternative.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'text' column.
    verbose : bool
        If True, downloads NLTK resources verbosely.

    Returns
    -------
    pd.DataFrame
        Copy of the input with normalized 'text'.
    """
    stop_words, lemmatizer = _ensure_nltk(verbose)

    url_pattern = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
    punct_table = str.maketrans("", "", string.punctuation)

    def normalize(s: str) -> str:
        s = s.lower()
        s = url_pattern.sub(" ", s)
        s = s.translate(punct_table)
        tokens = [t for t in s.split() if t and t.isalpha() and t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    out = df.copy()
    out["text"] = out["text"].astype(str).map(normalize)
    return out


# ---------- Class balancing and dataset builders ----------

def balance_classes(df: pd.DataFrame, label: str = "target", seed: int = 42) -> pd.DataFrame:
    """
    Naive upsampling with replacement until all classes match the majority count,
    then shuffle. Useful as a general utility, but DO NOT use it before the split
    (train/test or cross-validation) to avoid data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a label column.
    label : str
        Name of the label column.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Balanced dataframe (upsampled).
    """
    parts = []
    max_count = df[label].value_counts().max()
    for lab in df[label].unique():
        parts.append(
            resample(
                df[df[label] == lab],
                replace=True,
                n_samples=max_count,
                random_state=seed,
            )
        )
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_datasets(df: pd.DataFrame, cfg: Config):
    """
    Create train/test splits, optionally upsample ONLY the training split,
    build tf.data datasets, and adapt a TextVectorization layer on training text.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns 'text' (str) and 'target' (int).
    cfg : Config
        Configuration object; expects attributes:
          - test_size, random_state
          - no_balance (bool) to disable upsampling on train
          - vocab_size, seq_len, batch_size

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, tf.data.Dataset, tf.data.Dataset,
          tf.keras.layers.TextVectorization, int]
        X_test, y_test, train_ds, test_ds, vectorizer, num_classes
    """
    # Extract base arrays
    X = df["text"].values
    y = df["target"].values

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=getattr(cfg, "test_size", 0.2),
        random_state=getattr(cfg, "random_state", 42),
        stratify=y,
    )

    # Optional upsampling ONLY on the training data
    if not getattr(cfg, "no_balance", False):
        classes, counts = np.unique(y_train, return_counts=True)
        n_max = counts.max()
        Xb, yb = [], []
        for c in classes:
            Xi = X_train[y_train == c]
            yi = y_train[y_train == c]
            Xi_res, yi_res = resample(
                Xi,
                yi,
                replace=True,
                n_samples=n_max,
                random_state=getattr(cfg, "random_state", 42),
            )
            Xb.append(Xi_res)
            yb.append(yi_res)
        X_train = np.concatenate(Xb)
        y_train = np.concatenate(yb)

    # Text vectorizer adapted only on training text
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=getattr(cfg, "vocab_size", 20000),
        output_sequence_length=getattr(cfg, "seq_len", 128),
        output_mode="int",
    )
    ds_xtrain = tf.data.Dataset.from_tensor_slices(X_train).batch(getattr(cfg, "batch_size", 32))
    vectorizer.adapt(ds_xtrain)

    # Build tf.data datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(getattr(cfg, "buffer_size", 10000), seed=getattr(cfg, "random_state", 42))
        .batch(getattr(cfg, "batch_size", 32))
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(getattr(cfg, "batch_size", 32))
        .prefetch(tf.data.AUTOTUNE)
    )

    num_classes = int(np.unique(y).size)
    return X_test, y_test, train_ds, test_ds, vectorizer, num_classes
