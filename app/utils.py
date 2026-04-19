"""Helper functions for the Streamlit dashboard.

Kept deliberately small — most heavy lifting lives in ``src/``. This
module only handles UI concerns: cached loaders, colour mapping,
severity badges, and convenience Plotly builders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import BugSeverityPredictor


# Color mapping matches the ranking of severity — Blocker/Critical are red,
# Minor/Trivial green, Enhancement blue. The dashboard uses this in several
# places (badge, probability bar chart, pie chart) so it lives in one spot.
SEVERITY_COLORS: dict[str, str] = {
    "Blocker": "#b71c1c",
    "Critical": "#e53935",
    "Major": "#fb8c00",
    "Normal": "#fdd835",
    "Minor": "#43a047",
    "Trivial": "#2e7d32",
    "Enhancement": "#1e88e5",
}

SEVERITY_ORDER: list[str] = [
    "Blocker",
    "Critical",
    "Major",
    "Normal",
    "Minor",
    "Trivial",
    "Enhancement",
]


MODELS_DIR = Path("models")
METRICS_PATH = MODELS_DIR / "metrics.json"
TEST_SPLIT_PATH = MODELS_DIR / "test_split.csv"


def artifacts_available() -> bool:
    """True iff the minimum required training artifacts exist on disk."""
    required = [
        MODELS_DIR / "tfidf_vectorizer.pkl",
        MODELS_DIR / "label_encoder.pkl",
        MODELS_DIR / "logreg_model.pkl",
        MODELS_DIR / "rf_model.pkl",
    ]
    return all(p.exists() for p in required)


@st.cache_resource(show_spinner="Loading trained models ...")
def load_predictor() -> BugSeverityPredictor:
    """Load the predictor once and cache it across reruns."""
    return BugSeverityPredictor(MODELS_DIR)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    """Load the saved metrics.json; return {} if the file is missing."""
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str) -> pd.DataFrame | None:
    """Load a cleaned dataset CSV if it exists. Returns None on failure."""
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_test_split() -> pd.DataFrame | None:
    """Load the test split produced by training (for performance page)."""
    if not TEST_SPLIT_PATH.exists():
        return None
    return pd.read_csv(TEST_SPLIT_PATH)


def render_severity_badge(label: str) -> None:
    """Render a large colour-coded severity badge using HTML + inline CSS."""
    color = SEVERITY_COLORS.get(label, "#607d8b")
    html = f"""
    <div style="
        display:inline-block;
        padding:18px 36px;
        border-radius:12px;
        background:{color};
        color:white;
        font-size:28px;
        font-weight:700;
        letter-spacing:0.5px;
        box-shadow:0 4px 12px rgba(0,0,0,0.15);
    ">
        {label}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def probability_bar_chart(probabilities: dict[str, float]) -> go.Figure:
    """Horizontal Plotly bar chart of per-class probabilities."""
    ordered = [lbl for lbl in SEVERITY_ORDER if lbl in probabilities]
    values = [probabilities[lbl] for lbl in ordered]
    colors = [SEVERITY_COLORS.get(lbl, "#607d8b") for lbl in ordered]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=ordered,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=40, r=40, t=30, b=40),
        height=360,
        showlegend=False,
    )
    return fig


def severity_pie(df: pd.DataFrame, column: str = "severity") -> go.Figure:
    """Pie chart of class distribution."""
    counts = df[column].value_counts().reindex(SEVERITY_ORDER).dropna()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map=SEVERITY_COLORS,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def text_length_box(df: pd.DataFrame, text_col: str = "text") -> go.Figure:
    """Box plot of text character length grouped by severity."""
    work = df.copy()
    work["length"] = work[text_col].fillna("").astype(str).str.len()
    ordered = [s for s in SEVERITY_ORDER if s in work["severity"].unique()]
    fig = px.box(
        work,
        x="severity",
        y="length",
        color="severity",
        category_orders={"severity": ordered},
        color_discrete_map=SEVERITY_COLORS,
        points=False,
    )
    fig.update_layout(
        xaxis_title="Severity",
        yaxis_title="Text length (chars)",
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def component_severity_heatmap(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Heatmap of component (top N) by severity."""
    top_components = df["component"].value_counts().head(top_n).index
    pivot = (
        df.loc[df["component"].isin(top_components)]
        .groupby(["component", "severity"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[s for s in SEVERITY_ORDER if s in df["severity"].unique()])
        .fillna(0)
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(x="Severity", y="Component", color="Count"),
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def top_words_per_class(
    df: pd.DataFrame,
    clean_col: str = "clean_text",
    top_k: int = 10,
) -> dict[str, list[tuple[str, int]]]:
    """Return the top-k most frequent tokens per severity class.

    Expects ``clean_col`` to hold already-cleaned, space-separated tokens.
    Falls back to the ``text`` column if the cleaned column is missing.
    """
    from collections import Counter

    col = clean_col if clean_col in df.columns else "text"
    result: dict[str, list[tuple[str, int]]] = {}
    for severity, group in df.groupby("severity"):
        counter: Counter[str] = Counter()
        for txt in group[col].fillna(""):
            counter.update(str(txt).split())
        result[severity] = counter.most_common(top_k)
    return result


def confusion_heatmap(
    cm_values: list[list[int]],
    labels: list[str],
    title: str,
) -> go.Figure:
    """Plotly confusion matrix heatmap."""
    fig = px.imshow(
        cm_values,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        aspect="auto",
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=50, b=20))
    return fig
