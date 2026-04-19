"""Train bug-severity classifiers (Logistic Regression + Random Forest).

Usage:
    python src/train.py --data data/raw/bugs.csv
    python src/train.py --data data/raw/bugs.csv --smote
    python src/train.py --make-sample          # uses synthetic data

Produces the following artifacts under ``models/``:

- ``tfidf_vectorizer.pkl``
- ``label_encoder.pkl``
- ``logreg_model.pkl``
- ``rf_model.pkl``
- ``metrics.json``
- ``confusion_matrix_logreg.png``
- ``confusion_matrix_rf.png``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_bug_data, make_sample_dataset
from src.evaluate import (
    comparison_table,
    print_report,
    save_confusion_matrix,
)
from src.feature_engineering import fit_vectorizer
from src.preprocessor import clean_series


MODELS_DIR = Path("models")


def _maybe_apply_smote(
    X_train: Any,
    y_train: np.ndarray,
    use_smote: bool,
    labels: list[str],
) -> tuple[Any, np.ndarray]:
    """Optionally apply SMOTE to the training features.

    SMOTE's ``k_neighbors`` must be strictly smaller than the smallest
    class's sample count, so we auto-downshift ``k_neighbors`` when the
    data is sparse (common for the Trivial / Blocker classes).

    Args:
        X_train: Training feature matrix (sparse or dense).
        y_train: Encoded training labels.
        use_smote: Whether to apply SMOTE.
        labels: Ordered list of class names (for logging).

    Returns:
        ``(X_resampled, y_resampled)`` — identical to inputs when
        SMOTE is disabled or infeasible.
    """
    if not use_smote:
        return X_train, y_train

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[train] imbalanced-learn is not installed; skipping SMOTE.")
        return X_train, y_train

    counts = np.bincount(y_train)
    min_count = counts[counts > 0].min()
    if min_count < 2:
        print(
            f"[train] Smallest class has {min_count} samples; SMOTE needs >=2. "
            f"Skipping SMOTE."
        )
        return X_train, y_train

    k_neighbors = min(5, int(min_count) - 1)
    print(f"[train] Applying SMOTE (k_neighbors={k_neighbors})")
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    new_counts = np.bincount(y_res)
    print("[train] Post-SMOTE class counts:")
    for label, count in zip(labels, new_counts):
        print(f"    {label:<12s} {count:>7,d}")
    return X_res, y_res


def train_pipeline(
    csv_path: str | Path,
    models_dir: str | Path = MODELS_DIR,
    test_size: float = 0.2,
    use_smote: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the full training pipeline end-to-end.

    Args:
        csv_path: Path to the raw bug CSV (see ``data_loader`` for schema).
        models_dir: Directory where artifacts are written.
        test_size: Fraction of the dataset held out for evaluation.
        use_smote: If True, oversample minority classes in training set.
        random_state: Seed for reproducibility.

    Returns:
        A dict with metrics for each trained model.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[train] Loading data from {csv_path}")
    df = load_bug_data(csv_path)

    if df.empty:
        raise RuntimeError("No usable rows after cleaning — aborting training.")

    print("[train] Cleaning text ...")
    df["clean_text"] = clean_series(df["text"].tolist())
    df = df.loc[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"[train] {len(df):,} rows remain after text cleaning")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["severity"].values)
    labels: list[str] = list(label_encoder.classes_)
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    print(f"[train] Label classes: {labels}")

    # Stratified split. Using ``to_numpy(dtype=object)`` forces a plain
    # NumPy array — pandas 3.x returns a PyArrow-backed extension array
    # from ``.values`` for string columns, which sklearn's internal
    # fancy indexing can't handle.
    X_text_all = df["clean_text"].to_numpy(dtype=object)
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text_all,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"[train] Train size: {len(X_text_train):,} | Test size: {len(X_text_test):,}")

    # TF-IDF
    print("[train] Fitting TF-IDF vectorizer ...")
    vectorizer = fit_vectorizer(
        X_text_train, save_path=models_dir / "tfidf_vectorizer.pkl"
    )
    X_train = vectorizer.transform(X_text_train)
    X_test = vectorizer.transform(X_text_test)
    print(f"[train] TF-IDF matrix: {X_train.shape}")

    # Optional SMOTE
    X_train_res, y_train_res = _maybe_apply_smote(
        X_train, y_train, use_smote=use_smote, labels=labels
    )

    # --- Logistic Regression ---
    print("[train] Training Logistic Regression ...")
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
        random_state=random_state,
    )
    logreg.fit(X_train_res, y_train_res)
    logreg_pred = logreg.predict(X_test)
    logreg_metrics = print_report(
        "Logistic Regression", y_test, logreg_pred, labels=labels
    )
    joblib.dump(logreg, models_dir / "logreg_model.pkl")
    save_confusion_matrix(
        y_test,
        logreg_pred,
        labels=labels,
        out_path=models_dir / "confusion_matrix_logreg.png",
        title="Logistic Regression — Confusion Matrix",
    )

    # --- Random Forest ---
    print("[train] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train_res, y_train_res)
    rf_pred = rf.predict(X_test)
    rf_metrics = print_report("Random Forest", y_test, rf_pred, labels=labels)
    joblib.dump(rf, models_dir / "rf_model.pkl")
    save_confusion_matrix(
        y_test,
        rf_pred,
        labels=labels,
        out_path=models_dir / "confusion_matrix_rf.png",
        title="Random Forest — Confusion Matrix",
    )

    results = {
        "Logistic Regression": logreg_metrics,
        "Random Forest": rf_metrics,
    }

    # Persist metrics + test set for later dashboard use
    (models_dir / "metrics.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "n_train": int(len(X_text_train)),
                "n_test": int(len(X_text_test)),
                "use_smote": bool(use_smote),
                "results": {
                    k: {m: v for m, v in r.items()} for k, r in results.items()
                },
            },
            indent=2,
        )
    )
    # Save the cleaned test split so the dashboard's Model Performance
    # page can regenerate confusion matrices / ROC curves without reloading
    # the entire raw dataset.
    test_df = pd.DataFrame(
        {
            "clean_text": [str(t) if t is not None else "" for t in X_text_test],
            "severity_encoded": y_test,
            "severity": label_encoder.inverse_transform(y_test),
        }
    )
    # Quote every field so whitespace-only / empty strings survive the CSV
    # round-trip (otherwise pandas reads them back as NaN).
    import csv as _csv
    test_df.to_csv(
        models_dir / "test_split.csv", index=False, quoting=_csv.QUOTE_NONNUMERIC
    )

    print("\n[train] Model comparison:")
    print(comparison_table(results).round(4).to_string())

    print(f"\n[train] Done in {time.time() - t0:.1f}s. Artifacts in {models_dir}/")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bug severity classifiers")
    parser.add_argument(
        "--data",
        default="data/raw/bugs.csv",
        help="Path to raw bug report CSV (default: data/raw/bugs.csv)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help="Directory to write trained artifacts",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction"
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Use SMOTE to oversample minority classes in the training set",
    )
    parser.add_argument(
        "--make-sample",
        action="store_true",
        help=(
            "Generate a synthetic dataset at data/raw/sample.csv and train on it. "
            "Useful for verifying the pipeline without real data."
        ),
    )
    args = parser.parse_args()

    if args.make_sample:
        path = make_sample_dataset("data/raw/sample.csv")
        print(f"[train] Using synthetic sample dataset: {path}")
        train_pipeline(
            path,
            models_dir=args.models_dir,
            test_size=args.test_size,
            use_smote=args.smote,
        )
    else:
        train_pipeline(
            args.data,
            models_dir=args.models_dir,
            test_size=args.test_size,
            use_smote=args.smote,
        )


if __name__ == "__main__":
    main()
