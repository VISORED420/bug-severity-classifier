"""Unit tests for the text preprocessing pipeline."""

from __future__ import annotations

import pytest

from src.preprocessor import CUSTOM_STOPWORDS, clean_series, clean_text, strip_noise


def test_clean_text_handles_empty_and_non_string() -> None:
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]
    assert clean_text(123) == ""  # type: ignore[arg-type]


def test_strip_noise_removes_urls_emails_paths() -> None:
    raw = (
        "see https://bugs.eclipse.org/123 and contact dev@example.com. "
        "File c:\\workspace\\Main.java or /usr/local/bin/app failed."
    )
    stripped = strip_noise(raw)
    assert "https" not in stripped
    assert "@" not in stripped
    assert "Main.java" not in stripped
    assert "/usr/local" not in stripped


def test_strip_noise_removes_stack_traces() -> None:
    raw = (
        "NullPointerException at com.example.Foo.bar(Foo.java:42) occurred "
        "and then IllegalStateException bubbled up."
    )
    stripped = strip_noise(raw)
    assert "Foo.java" not in stripped
    assert "NullPointerException" not in stripped
    assert "IllegalStateException" not in stripped


def test_clean_text_lowercases_and_removes_special_chars() -> None:
    cleaned = clean_text("Serious CRASH!!! @#$%")
    assert cleaned == cleaned.lower()
    assert "@" not in cleaned
    assert "#" not in cleaned


def test_clean_text_removes_stopwords_and_custom_stopwords() -> None:
    cleaned = clean_text("This bug is a serious issue with the editor")
    tokens = cleaned.split()
    # Custom tech stopwords should be gone
    assert "bug" not in tokens
    assert "issue" not in tokens
    # Common English stopwords should be gone
    assert "is" not in tokens
    assert "a" not in tokens
    assert "the" not in tokens
    # Content words survive
    assert "serious" in tokens
    assert "editor" in tokens


def test_clean_text_lemmatizes_plurals() -> None:
    cleaned = clean_text("crashes crashes crashes")
    # "crashes" -> "crash" via WordNet
    assert "crash" in cleaned.split()


def test_clean_series_returns_same_length() -> None:
    texts = ["a crash in the editor", "", "tool tip alignment off"]
    out = clean_series(texts)
    assert len(out) == len(texts)
    assert out[1] == ""


def test_custom_stopwords_contains_expected() -> None:
    assert "bug" in CUSTOM_STOPWORDS
    assert "issue" in CUSTOM_STOPWORDS
    assert "eclipse" in CUSTOM_STOPWORDS


def test_clean_text_drops_very_short_tokens() -> None:
    # Tokens shorter than min_token_length (default 2) should be dropped.
    cleaned = clean_text("a b c important")
    assert "a" not in cleaned.split()
    assert "important" in cleaned.split()


def test_clean_text_extra_stopwords() -> None:
    cleaned = clean_text("debugger missing breakpoint", extra_stopwords={"debugger"})
    tokens = cleaned.split()
    assert "debugger" not in tokens
    assert "breakpoint" in tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
