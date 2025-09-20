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

# NLTK para stopwords en español (ligero y fácil de instalar)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .config import Config


# ---------- Carga y normalización de columnas ----------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Carga CSV y normaliza a ('text','target').
    - Autodetecta separador y encoding.
    - Acepta alias comunes en inglés/español.
    - Normaliza etiquetas a enteros 0..C-1.
    """
    import pandas as pd

    attempts = [
        ("utf-8", ";"),
        ("utf-8", ","),
        ("utf-8", None),     # inferir
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

    # normaliza nombres
    df.columns = [str(c).strip().lower() for c in df.columns]

    # alias de texto y etiqueta (EN + ES)
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

    df = df.rename(columns={text_col: "text", target_col: "target"}).copy()

    # descarta columnas auxiliares comunes si aparecen
    for col in ("id", "tr", "ag"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # normaliza etiquetas a 0..C-1 enteros
    df["target"] = pd.Categorical(df["target"]).codes.astype(int)

    return df[["text", "target"]]

# ---------- Preprocesado de texto ----------

def _ensure_nltk(verbose: bool = False) -> tuple[set[str], WordNetLemmatizer]:
    """
    Garantiza que los recursos de NLTK necesarios están disponibles y devuelve
    stopwords españolas + lematizador.
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
    Normaliza texto: minúsculas, elimina URLs y puntuación, quita stopwords en español
    y lematiza con WordNet (nota: lematiza mejor en inglés; para español óptimo,
    considerar spaCy 'es_core_news_sm').
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


# ---------- Balanceo y datasets ----------

def balance_classes(df: pd.DataFrame, label: str = "target", seed: int = 42) -> pd.DataFrame:
    """
    Upsampling con reemplazo hasta igualar la clase mayoritaria y barajado final.
    ÚTIL solo como utilidad general; NO usar antes del split para evitar leakage.
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
    Divide en train/test estratificado, opcionalmente balancea SOLO el train,
    crea tf.data Datasets y una capa TextVectorization adaptada sobre train.

    Devuelve:
      X_test (np.ndarray[str]), y_test (np.ndarray[int]),
      train_ds (tf.data.Dataset), test_ds (tf.data.Dataset),
      vectorizer (tf.keras.layers.TextVectorization), num_classes (int)
    """
    # Extrae arrays base
    X = df["text"].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=getattr(cfg, "test_size", 0.2),
        random_state=getattr(cfg, "random_state", 42),
        stratify=y,
    )

    # Balanceo SOLO en train si no se desactiva
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

    # Vectorizer adaptado solo con X_train
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=getattr(cfg, "vocab_size", 20000),
        output_sequence_length=getattr(cfg, "seq_len", 128),
        output_mode="int",
    )
    ds_xtrain = tf.data.Dataset.from_tensor_slices(X_train).batch(getattr(cfg, "batch_size", 32))
    vectorizer.adapt(ds_xtrain)

    # tf.data Datasets
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
