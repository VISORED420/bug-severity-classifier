"""Convert the MSR-style Eclipse Bugzilla JSON dump into a flat CSV.

The MSR (Mining Software Repositories) Eclipse dataset stores bug data
as a set of per-column JSON files. Each file looks like:

    {"<column>": {"<bug_id>": [{"when": <ts>, "what": <value>, "who": <id>}, ...], ...}}

i.e. a dict keyed by bug_id where each value is the change-history of that
column. The **latest** entry in the list is the current value.

The ``reports.json`` file is structured slightly differently — each bug_id
maps directly to a dict with ``opening``, ``reporter``, ``current_status``
and ``current_resolution`` fields.

This module extracts the *current* value of each column and writes a
single CSV with the schema expected by :mod:`src.data_loader`.

Usage::

    python -m src.msr_adapter --raw-dir data/raw --out data/raw/bugs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


HISTORY_COLUMNS: tuple[str, ...] = (
    "severity",
    "short_desc",
    "component",
    "priority",
    "bug_status",
    "resolution",
    "product",
    "op_sys",
    "version",
    "assigned_to",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return its top-level dict payload."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_value(history: Any) -> str | None:
    """Return the most recent ``what`` value from a change-history list.

    Args:
        history: Either a list of ``{"when": ..., "what": ..., "who": ...}``
            dicts or a bare scalar.

    Returns:
        The latest value as a string, or ``None`` if history is empty.
    """
    if not isinstance(history, list) or not history:
        return None
    # Sort by `when` to pick the latest, defending against unordered dumps.
    try:
        latest = max(history, key=lambda x: x.get("when", 0))
    except (TypeError, AttributeError):
        latest = history[-1]
    value = latest.get("what") if isinstance(latest, dict) else latest
    return None if value is None else str(value)


def convert_msr_dump(raw_dir: str | Path, out_csv: str | Path) -> Path:
    """Convert a directory of MSR Eclipse JSON files into a single CSV.

    The resulting CSV has the columns expected by
    :func:`src.data_loader.load_bug_data`:

    - ``bug_id``, ``summary``, ``description``, ``severity``,
      ``component``, ``priority``.

    Because the MSR dump only contains the short description (no full
    multi-comment body), ``summary`` is populated from ``short_desc`` and
    ``description`` is left empty — ``data_loader`` happily accepts that.

    Args:
        raw_dir: Directory containing the per-column JSON files.
        out_csv: Destination CSV path.

    Returns:
        Resolved output path.

    Raises:
        FileNotFoundError: If required source JSON files are missing.
    """
    raw_dir = Path(raw_dir)
    out_csv = Path(out_csv)

    required = ["severity.json", "short_desc.json", "component.json", "priority.json"]
    missing = [f for f in required if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing MSR JSON files in {raw_dir}: {missing}"
        )

    # Load per-column history dumps. Each file wraps its payload in a
    # single top-level key matching the filename.
    histories: dict[str, dict[str, Any]] = {}
    for col in HISTORY_COLUMNS:
        path = raw_dir / f"{col}.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        # Payload is {"<col>": {...}}
        inner = payload.get(col, payload)
        if not isinstance(inner, dict):
            continue
        histories[col] = inner

    # Also load reports.json — it's structured differently and gives us
    # the opening timestamp and reporter, which we don't directly need
    # for classification but which help produce a complete CSV.
    reports_path = raw_dir / "reports.json"
    reports: dict[str, dict[str, Any]] = {}
    if reports_path.exists():
        payload = _load_json(reports_path)
        inner = payload.get("reports", payload)
        if isinstance(inner, dict):
            reports = inner

    # Union of all bug_ids seen across files (prefer short_desc as the
    # anchor since a row with no summary is useless).
    anchor_ids = set(histories.get("short_desc", {}).keys())
    if not anchor_ids:
        anchor_ids = set(reports.keys())
    print(f"[msr_adapter] {len(anchor_ids):,} unique bug_ids to convert")

    rows: list[dict[str, Any]] = []
    for bug_id in anchor_ids:
        summary = _latest_value(histories.get("short_desc", {}).get(bug_id))
        if not summary:
            continue
        row: dict[str, Any] = {
            "bug_id": bug_id,
            "summary": summary,
            "description": "",  # MSR dump has no full description
            "severity": _latest_value(histories.get("severity", {}).get(bug_id))
            or "normal",
            "component": _latest_value(histories.get("component", {}).get(bug_id))
            or "Unknown",
            "priority": _latest_value(histories.get("priority", {}).get(bug_id))
            or "P3",
            "bug_status": _latest_value(histories.get("bug_status", {}).get(bug_id))
            or "",
            "resolution": _latest_value(histories.get("resolution", {}).get(bug_id))
            or "",
            "product": _latest_value(histories.get("product", {}).get(bug_id)) or "",
        }
        report = reports.get(bug_id)
        if isinstance(report, dict):
            row["opening"] = report.get("opening", "")
            row["reporter"] = report.get("reporter", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[msr_adapter] Wrote {len(df):,} rows to {out_csv}")
    print("[msr_adapter] Severity distribution in output:")
    print(df["severity"].value_counts().to_string())
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MSR Eclipse JSON dump into data_loader-compatible CSV."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing severity.json, short_desc.json, ...",
    )
    parser.add_argument(
        "--out",
        default="data/raw/bugs.csv",
        help="Destination CSV path",
    )
    args = parser.parse_args()
    convert_msr_dump(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
