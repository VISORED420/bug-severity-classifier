"""TF-IDF feature extraction for bug report text."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")


def build_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """Construct the standard TF-IDF vectorizer used throughout the project.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: Range of n-gram sizes to extract.
        min_df: Ignore terms appearing in fewer than this many documents.
        max_df: Ignore terms appearing in more than this fraction of documents.

    Returns:
        An unfitted ``TfidfVectorizer`` with the requested configuration.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def fit_vectorizer(
    texts: Iterable[str],
    save_path: str | Path | None = DEFAULT_VECTORIZER_PATH,
    **kwargs,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on ``texts`` and optionally persist it.

    Args:
        texts: Iterable of cleaned bug texts.
        save_path: If provided, the fitted vectorizer is written here with
            :func:`joblib.dump`. Pass ``None`` to skip saving.
        **kwargs: Forwarded to :func:`build_vectorizer`.

    Returns:
        The fitted vectorizer.
    """
    vectorizer = build_vectorizer(**kwargs)
    vectorizer.fit(list(texts))
    if save_path is not None:
        save_vectorizer(vectorizer, save_path)
    return vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> Path:
    """Persist a fitted vectorizer to disk.

    Args:
        vectorizer: A fitted ``TfidfVectorizer``.
        path: Destination path (parent dirs are created).

    Returns:
        The resolved path where the vectorizer was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(path: str | Path = DEFAULT_VECTORIZER_PATH) -> TfidfVectorizer:
    """Load a previously saved vectorizer.

    Args:
        path: Path to the pickled vectorizer.

    Returns:
        The deserialised ``TfidfVectorizer``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {path}")
    return joblib.load(path)
