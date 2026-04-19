"""Load and clean Eclipse/Bugzilla bug report data.

This module handles loading raw bug report CSV files, filtering to
supported severity classes, merging summary + description into a single
text field, and printing class distribution for imbalance analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


VALID_SEVERITIES: tuple[str, ...] = (
    "Blocker",
    "Critical",
    "Major",
    "Normal",
    "Minor",
    "Trivial",
    "Enhancement",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "bug_id",
    "summary",
    "description",
    "severity",
    "component",
    "priority",
)


def _normalize_severity(value: object) -> str | None:
    """Normalize severity string to title-case canonical form.

    Bugzilla exports often use lowercase values (e.g. ``critical``,
    ``enhancement``); this helper maps any case variant to the canonical
    title-cased label used by the rest of the pipeline.

    Args:
        value: Raw severity cell from the source CSV.

    Returns:
        The canonical severity label, or ``None`` if the input does not
        correspond to a supported class.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().title()
    return cleaned if cleaned in VALID_SEVERITIES else None


def load_bug_data(
    csv_path: str | Path,
    valid_severities: Iterable[str] = VALID_SEVERITIES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load bug report data from a CSV file and clean it.

    The loader performs the following steps:

    1. Reads the CSV, coercing missing values to NaN.
    2. Verifies that required columns are present (missing optional
       columns are filled with empty strings so the pipeline stays
       robust against minor schema variation).
    3. Drops rows where BOTH ``summary`` and ``description`` are null
       (a row with no text cannot be classified).
    4. Normalizes and filters severity values to the supported classes.
    5. Combines ``summary`` and ``description`` into a ``text`` column.
    6. Prints a class distribution table when ``verbose`` is True.

    Args:
        csv_path: Path to the raw bug report CSV.
        valid_severities: Iterable of allowed severity labels.
        verbose: Whether to print dataset statistics.

    Returns:
        A cleaned ``DataFrame`` with an added ``text`` column.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If any required column is missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Bug data CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {missing}. "
            f"Expected: {list(REQUIRED_COLUMNS)}"
        )

    initial_rows = len(df)

    df["summary"] = df["summary"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["component"] = df["component"].fillna("Unknown").astype(str)
    df["priority"] = df["priority"].fillna("Unknown").astype(str)

    # Drop rows where both text fields are empty
    both_empty = (df["summary"].str.strip() == "") & (
        df["description"].str.strip() == ""
    )
    df = df.loc[~both_empty].copy()

    # Normalize and filter severity
    df["severity"] = df["severity"].apply(_normalize_severity)
    allowed = set(valid_severities)
    df = df.loc[df["severity"].isin(allowed)].copy()

    # Combine summary + description into a single text field
    df["text"] = (df["summary"].str.strip() + " " + df["description"].str.strip()).str.strip()

    df = df.reset_index(drop=True)

    if verbose:
        print(f"[data_loader] Loaded {initial_rows:,} raw rows from {csv_path.name}")
        print(f"[data_loader] After cleaning: {len(df):,} rows")
        print("[data_loader] Severity class distribution:")
        dist = df["severity"].value_counts()
        total = dist.sum()
        for label, count in dist.items():
            pct = 100.0 * count / total if total else 0.0
            print(f"    {label:<12s} {count:>7,d}  ({pct:5.2f}%)")

    return df


def save_processed(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write the cleaned DataFrame to ``out_path`` as CSV.

    Args:
        df: Cleaned DataFrame returned by :func:`load_bug_data`.
        out_path: Destination CSV path; parent dirs are created.

    Returns:
        The resolved output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def make_sample_dataset(out_path: str | Path, n_per_class: int = 40) -> Path:
    """Create a small synthetic dataset for smoke testing the pipeline.

    This is useful when an Eclipse/Bugzilla export is not yet available
    locally; it lets you run ``train.py`` end-to-end to verify wiring.

    Args:
        out_path: Where to write the generated CSV.
        n_per_class: Number of synthetic rows per severity class.

    Returns:
        The resolved output path.
    """
    import random

    random.seed(42)

    templates: dict[str, list[str]] = {
        "Blocker": [
            "Application crashes on startup with NullPointerException",
            "Entire build fails, cannot compile the project",
            "IDE freezes completely when opening any workspace",
        ],
        "Critical": [
            "Data loss when saving large files in editor",
            "Memory leak causes out of memory after long session",
            "Critical security vulnerability in login handler",
        ],
        "Major": [
            "Incorrect compilation error shown for valid Java code",
            "Refactoring tool renames unrelated variables",
            "Debugger fails to hit breakpoints in inner classes",
        ],
        "Normal": [
            "Syntax highlighting wrong for multi-line strings",
            "Autocomplete sometimes misses imports",
            "Wizard dialog has minor layout issues on resize",
        ],
        "Minor": [
            "Tooltip shows truncated text for long method names",
            "Icon alignment off by a pixel in toolbar",
            "Status bar text flickers briefly on save",
        ],
        "Trivial": [
            "Typo in preferences dialog label",
            "Inconsistent capitalization in menu items",
            "Extra whitespace in generated comment header",
        ],
        "Enhancement": [
            "Add dark theme option for editor",
            "Support exporting project settings to JSON",
            "Provide keyboard shortcut for split editor",
        ],
    }

    components = ["UI", "Core", "Debug", "Editor", "Build", "Search"]
    priorities = ["P1", "P2", "P3", "P4", "P5"]

    rows: list[dict[str, object]] = []
    bug_id = 1000
    for severity, texts in templates.items():
        for i in range(n_per_class):
            base = random.choice(texts)
            suffix = f" (case #{i})"
            summary = base + suffix
            description = (
                f"Steps to reproduce: open the IDE, perform action {i}. "
                f"Expected: normal behaviour. Actual: {base.lower()}."
            )
            rows.append(
                {
                    "bug_id": bug_id,
                    "summary": summary,
                    "description": description,
                    "severity": severity,
                    "component": random.choice(components),
                    "priority": random.choice(priorities),
                }
            )
            bug_id += 1

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Eclipse/Bugzilla bug data")
    parser.add_argument("csv_path", nargs="?", help="Path to raw bug CSV")
    parser.add_argument(
        "--make-sample",
        action="store_true",
        help="Generate a small synthetic dataset under data/raw/sample.csv",
    )
    args = parser.parse_args()

    if args.make_sample:
        path = make_sample_dataset("data/raw/sample.csv")
        print(f"Wrote synthetic sample dataset to {path}")
    elif args.csv_path:
        load_bug_data(args.csv_path)
    else:
        parser.print_help()
