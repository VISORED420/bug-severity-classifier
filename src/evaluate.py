"""Evaluation utilities: metrics, classification reports, confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe (training may run without display)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1 (macro + weighted) and per-class F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional ordered list of class names.

    Returns:
        Dict with scalar metrics and a ``per_class_f1`` mapping.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    per_class = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    if labels is not None:
        metrics["per_class_f1"] = {
            lbl: float(score) for lbl, score in zip(labels, per_class)
        }
    else:
        metrics["per_class_f1"] = {
            f"class_{i}": float(score) for i, score in enumerate(per_class)
        }
    return metrics


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> str:
    """Return sklearn's classification report as a formatted string."""
    return classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Render a confusion matrix heatmap and write it to ``out_path`` as PNG.

    Args:
        y_true: Ground-truth labels (encoded or string).
        y_pred: Predicted labels (same encoding as ``y_true``).
        labels: Ordered list of class names matching the encoding.
        out_path: Destination PNG path.
        title: Figure title.

    Returns:
        The resolved output path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def comparison_table(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a side-by-side comparison table from a results dict.

    Args:
        results: Mapping of model name -> metrics dict from
            :func:`compute_metrics`.

    Returns:
        A DataFrame indexed by metric with one column per model.
    """
    scalar_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    rows: dict[str, dict[str, float]] = {k: {} for k in scalar_keys}
    for model_name, metrics in results.items():
        for k in scalar_keys:
            rows[k][model_name] = metrics.get(k, float("nan"))
    return pd.DataFrame(rows).T


def print_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Print a full evaluation report for one model and return metrics.

    Args:
        model_name: Display name for the model.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of class names.

    Returns:
        The metrics dict from :func:`compute_metrics`.
    """
    print(f"\n=== {model_name} ===")
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 (macro): {metrics['f1_macro']:.4f} | "
        f"F1 (weighted): {metrics['f1_weighted']:.4f}"
    )
    print("\nClassification report:")
    print(classification_report_text(y_true, y_pred, labels=labels))
    return metrics
