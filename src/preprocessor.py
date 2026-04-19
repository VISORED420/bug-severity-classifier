"""Text preprocessing for bug report summaries and descriptions.

Provides a small, dependency-light text cleaning pipeline used both at
training time and at inference time inside the Streamlit app.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Tech / project-specific stop tokens that add noise to bug text.
CUSTOM_STOPWORDS: frozenset[str] = frozenset(
    {
        "bug",
        "issue",
        "problem",
        "report",
        "reported",
        "please",
        "eclipse",
        "ide",
        "version",
        "user",
        "users",
        "thanks",
        "thank",
        "hi",
        "hello",
        "also",
        "would",
        "could",
        "should",
        "like",
        "one",
        "two",
        "get",
        "got",
        "see",
        "seen",
    }
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Windows + POSIX paths (c:\foo\bar, /usr/local/foo, ./rel/path)
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|[\\/])\S+")
# Heuristic stack-trace marker: "at com.foo.Bar.baz(Bar.java:42)"
_STACK_AT_RE = re.compile(r"\bat\s+[\w.$]+\([^\)]*\)")
_STACK_EXC_RE = re.compile(r"\b[\w.$]+(?:Exception|Error)\b:?")
# Collapse anything that isn't alphanumeric or basic punctuation
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s.,!?'-]")
_WHITESPACE_RE = re.compile(r"\s+")


def _ensure_nltk_resources() -> None:
    """Download NLTK resources on first use (idempotent, quiet)."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                # punkt_tab only exists in newer NLTK; ignore failures
                pass


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    _ensure_nltk_resources()
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    _ensure_nltk_resources()
    try:
        base = set(nltk_stopwords.words("english"))
    except LookupError:
        base = set()
    return frozenset(base | CUSTOM_STOPWORDS)


def strip_noise(text: str) -> str:
    """Remove URLs, emails, file paths, and stack traces.

    Args:
        text: Raw input text.

    Returns:
        Text with noisy, high-cardinality tokens removed.
    """
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PATH_RE.sub(" ", text)
    text = _STACK_AT_RE.sub(" ", text)
    text = _STACK_EXC_RE.sub(" ", text)
    return text


def _simple_tokenize(text: str) -> list[str]:
    """Tokenize with NLTK when available, fall back to whitespace split."""
    try:
        return word_tokenize(text)
    except LookupError:
        _ensure_nltk_resources()
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()


def clean_text(
    text: str,
    extra_stopwords: Iterable[str] | None = None,
    min_token_length: int = 2,
) -> str:
    """Run the full cleaning pipeline on a single bug text.

    The pipeline performs the following steps in order:

    1. Null/empty handling
    2. Lowercasing
    3. Noise stripping (URLs, emails, paths, stack traces)
    4. Special character removal (keeps alphanumerics + basic punctuation)
    5. Tokenization
    6. Stopword removal (NLTK English + custom tech tokens)
    7. Lemmatization

    Args:
        text: Raw text to clean.
        extra_stopwords: Additional stopwords to drop in this call.
        min_token_length: Minimum length a token must have to be kept.

    Returns:
        The cleaned text as a single space-joined string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = strip_noise(text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    tokens = _simple_tokenize(text)
    stops = set(_get_stopwords())
    if extra_stopwords:
        stops.update(s.lower() for s in extra_stopwords)

    lemmatizer = _get_lemmatizer()
    cleaned: list[str] = []
    for tok in tokens:
        if len(tok) < min_token_length:
            continue
        if not tok.isalnum():
            continue
        if tok in stops:
            continue
        lemma = lemmatizer.lemmatize(tok)
        if lemma in stops or len(lemma) < min_token_length:
            continue
        cleaned.append(lemma)

    return " ".join(cleaned)


def clean_series(texts: Iterable[str]) -> list[str]:
    """Apply :func:`clean_text` to an iterable of texts.

    Args:
        texts: Iterable of raw bug texts.

    Returns:
        A list of cleaned strings of the same length as the input.
    """
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    sample = (
        "NullPointerException at com.example.Foo.bar(Foo.java:42) "
        "please visit https://bugs.eclipse.org/123 for details. "
        "File c:\\workspace\\Main.java has the issue."
    )
    print("Raw:    ", sample)
    print("Cleaned:", clean_text(sample))
