"""Inference pipeline for bug severity classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from src.feature_engineering import load_vectorizer
from src.preprocessor import clean_text


MODELS_DIR = Path("models")

MODEL_FILES: dict[str, str] = {
    "Logistic Regression": "logreg_model.pkl",
    "Random Forest": "rf_model.pkl",
}


@dataclass
class Prediction:
    """Structured result of a single prediction call."""

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    cleaned_text: str = ""


class BugSeverityPredictor:
    """Thin wrapper around vectorizer + classifier + label encoder.

    Example:
        >>> predictor = BugSeverityPredictor()
        >>> predictor.predict("App crashes on startup", model_name="Logistic Regression")
        Prediction(label='Blocker', confidence=0.42, ...)
    """

    def __init__(self, models_dir: str | Path = MODELS_DIR) -> None:
        """Load all persisted artifacts.

        Args:
            models_dir: Directory containing the saved ``.pkl`` artifacts
                produced by :func:`src.train.train_pipeline`.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        self.models_dir = Path(models_dir)
        self.vectorizer = load_vectorizer(self.models_dir / "tfidf_vectorizer.pkl")

        encoder_path = self.models_dir / "label_encoder.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
        self.label_encoder = joblib.load(encoder_path)
        self.labels: list[str] = list(self.label_encoder.classes_)

        self.models: dict[str, object] = {}
        for name, filename in MODEL_FILES.items():
            path = self.models_dir / filename
            if path.exists():
                self.models[name] = joblib.load(path)
        if not self.models:
            raise FileNotFoundError(
                f"No model files found in {self.models_dir}. "
                f"Expected one of: {list(MODEL_FILES.values())}"
            )

    @property
    def available_models(self) -> list[str]:
        """Names of loaded models."""
        return list(self.models)

    def predict(
        self,
        text: str,
        model_name: str = "Logistic Regression",
    ) -> Prediction:
        """Predict severity for a single bug text.

        Args:
            text: Raw bug text (summary + description).
            model_name: Which model to use. Must be one of
                :attr:`available_models`.

        Returns:
            A :class:`Prediction` with label, confidence and per-class
            probabilities.

        Raises:
            KeyError: If ``model_name`` was not loaded.
            ValueError: If ``text`` is empty after cleaning.
        """
        if model_name not in self.models:
            raise KeyError(
                f"Model '{model_name}' not available. "
                f"Available: {self.available_models}"
            )

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Input text is empty after preprocessing.")

        X = self.vectorizer.transform([cleaned])
        model = self.models[model_name]

        probs: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        else:
            pred_idx = int(model.predict(X)[0])
            confidence = 1.0

        label = self.labels[pred_idx]
        prob_map: dict[str, float] = {}
        if probs is not None:
            prob_map = {lbl: float(p) for lbl, p in zip(self.labels, probs)}
        return Prediction(
            label=label,
            confidence=confidence,
            probabilities=prob_map,
            cleaned_text=cleaned,
        )

    def top_influential_words(
        self,
        text: str,
        top_k: int = 10,
        model_name: str = "Logistic Regression",
    ) -> list[tuple[str, float]]:
        """Return the top words driving a Logistic Regression prediction.

        This computes ``tfidf_value_i * coefficient_i`` for each feature
        present in the input text, using the coefficient row for the
        predicted class. Works only for Logistic Regression (a linear
        model with interpretable coefficients).

        Args:
            text: Raw bug text.
            top_k: How many top-contributing tokens to return.
            model_name: Must be ``"Logistic Regression"``.

        Returns:
            A list of ``(token, contribution)`` pairs sorted descending
            by absolute contribution. Empty if unsupported model or
            empty input.
        """
        if model_name != "Logistic Regression":
            return []
        if model_name not in self.models:
            return []

        cleaned = clean_text(text)
        if not cleaned:
            return []

        X = self.vectorizer.transform([cleaned])
        model = self.models[model_name]
        if not hasattr(model, "coef_"):
            return []

        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        coefs = model.coef_[pred_idx]

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        x_arr = X.toarray()[0]
        contributions = x_arr * coefs
        nonzero = np.nonzero(x_arr)[0]
        if nonzero.size == 0:
            return []

        ranked = sorted(
            ((feature_names[i], float(contributions[i])) for i in nonzero),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        return ranked[:top_k]


if __name__ == "__main__":
    import sys

    predictor = BugSeverityPredictor()
    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])
    else:
        sample = "Application crashes on startup with NullPointerException"
    print(f"Input:  {sample}")
    result = predictor.predict(sample, model_name="Logistic Regression")
    print(f"Label:      {result.label}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Top tokens: {predictor.top_influential_words(sample)}")
