"""Bug Severity Classifier — Streamlit dashboard.

Run with:
    streamlit run app/streamlit_app.py

The dashboard has four pages, selectable from the sidebar:
    * Home              — project overview, dataset stats, model performance cards
    * Predict Severity  — interactive prediction for user-entered text
    * Data Insights     — class distribution, word frequencies, length, heatmap
    * Model Performance — confusion matrices, metrics table, ROC curves
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

# Ensure project root is importable when running `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import (  # noqa: E402
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    artifacts_available,
    component_severity_heatmap,
    confusion_heatmap,
    load_dataset,
    load_metrics,
    load_predictor,
    load_test_split,
    probability_bar_chart,
    render_severity_badge,
    severity_pie,
    text_length_box,
    top_words_per_class,
)


st.set_page_config(
    page_title="Bug Severity Classifier",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS — most colouring is inline on the badges themselves.
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; }
    .metric-card {
        background: #f5f7fa;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🏠 Home": "home",
    "🔮 Predict Severity": "predict",
    "📊 Data Insights": "insights",
    "📈 Model Performance": "performance",
}

st.sidebar.title("🐞 Bug Severity Classifier")
page_label = st.sidebar.radio("Navigate", list(PAGES.keys()))
page = PAGES[page_label]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Classical ML (TF-IDF + Logistic Regression / Random Forest) "
    "for classifying Eclipse-style bug reports."
)

# ---------------------------------------------------------------------------
# Artifact guard
# ---------------------------------------------------------------------------

artifacts_ok = artifacts_available()
if not artifacts_ok and page in ("predict", "performance"):
    st.warning(
        "Trained models were not found in `models/`. "
        "Train them first by running:\n\n"
        "```bash\npython -m src.train --make-sample\n```\n\n"
        "or, with your own CSV:\n\n"
        "```bash\npython -m src.train --data data/raw/your_bugs.csv\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def render_home() -> None:
    st.title("🐞 Bug Severity Classifier")
    st.markdown(
        "An interactive ML dashboard that classifies bug reports into "
        "**seven severity levels** — from *Blocker* to *Enhancement* — using "
        "TF-IDF features and two classical models for comparison."
    )

    metrics = load_metrics()
    results = metrics.get("results", {}) if metrics else {}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 Dataset")
        if metrics:
            total = metrics.get("n_train", 0) + metrics.get("n_test", 0)
            st.metric("Total examples", f"{total:,}")
            st.metric("Classes", len(metrics.get("labels", [])))
        else:
            st.info("Train a model to populate dataset stats.")

    with col2:
        st.markdown("### 🤖 Logistic Regression")
        lr = results.get("Logistic Regression")
        if lr:
            st.metric("Accuracy", f"{lr['accuracy']:.2%}")
            st.metric("F1 (macro)", f"{lr['f1_macro']:.3f}")
            st.metric("F1 (weighted)", f"{lr['f1_weighted']:.3f}")
        else:
            st.info("No metrics yet.")

    with col3:
        st.markdown("### 🌲 Random Forest")
        rf = results.get("Random Forest")
        if rf:
            st.metric("Accuracy", f"{rf['accuracy']:.2%}")
            st.metric("F1 (macro)", f"{rf['f1_macro']:.3f}")
            st.metric("F1 (weighted)", f"{rf['f1_weighted']:.3f}")
        else:
            st.info("No metrics yet.")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
1. **Load & clean** Eclipse/Bugzilla bug reports (CSV with `summary`, `description`, `severity`, ...).
2. **Preprocess** text — remove URLs/emails/stack traces, lowercase, strip stopwords, lemmatize.
3. **Vectorize** with TF-IDF (unigrams + bigrams, up to 5,000 features).
4. **Train** Logistic Regression and Random Forest with balanced class weights (optional SMOTE).
5. **Evaluate** on a stratified hold-out set and compare metrics.
6. **Explore** predictions and dataset statistics in this dashboard.
        """
    )

    if not artifacts_ok:
        st.info(
            "👈 Train a model to unlock the **Predict** and **Model Performance** pages.\n\n"
            "```bash\npython -m src.train --make-sample\n```"
        )


# ---------------------------------------------------------------------------
# Page: Predict
# ---------------------------------------------------------------------------

def render_predict() -> None:
    st.title("🔮 Predict Bug Severity")
    st.markdown("Enter a bug report summary and description to classify it.")

    predictor = load_predictor()
    available_models = predictor.available_models

    with st.form("predict_form"):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            summary = st.text_input(
                "Summary",
                placeholder="e.g. IDE crashes on startup when opening workspace",
            )
            description = st.text_area(
                "Description",
                height=160,
                placeholder=(
                    "Steps to reproduce, expected vs actual behaviour, "
                    "stack traces, environment info ..."
                ),
            )
        with col_b:
            model_name = st.selectbox("Model", available_models, index=0)
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if not submitted:
        return

    combined = f"{summary.strip()} {description.strip()}".strip()
    if not combined:
        st.error("Please enter a summary and/or description.")
        return

    try:
        with st.spinner("Classifying ..."):
            result = predictor.predict(combined, model_name=model_name)
    except ValueError:
        st.error("After preprocessing, the input contained no usable tokens. Add more detail.")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.markdown("### Prediction")
    col1, col2 = st.columns([1, 2])
    with col1:
        render_severity_badge(result.label)
        st.markdown(f"**Confidence:** {result.confidence:.1%}")
        st.progress(min(max(result.confidence, 0.0), 1.0))
    with col2:
        if result.probabilities:
            st.plotly_chart(
                probability_bar_chart(result.probabilities),
                use_container_width=True,
            )

    if model_name == "Logistic Regression":
        st.markdown("### Top influential words")
        words = predictor.top_influential_words(combined, top_k=10, model_name=model_name)
        if words:
            wdf = pd.DataFrame(words, columns=["Token", "Contribution"])
            fig = px.bar(
                wdf,
                x="Contribution",
                y="Token",
                orientation="h",
                color="Contribution",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Positive (blue) = pushes toward the predicted class; "
                "negative (red) = pushes away."
            )
        else:
            st.info("No influential tokens available for this input.")

    with st.expander("🔍 Cleaned text used for prediction"):
        st.code(result.cleaned_text or "(empty)")


# ---------------------------------------------------------------------------
# Page: Data Insights
# ---------------------------------------------------------------------------

def render_insights() -> None:
    st.title("📊 Data Insights")

    # Prefer the processed dataset if present; fall back to raw sample.
    df = load_dataset("data/processed/bugs_clean.csv")
    if df is None:
        df = load_dataset("data/raw/sample.csv")
    if df is None:
        df = load_dataset("data/raw/bugs.csv")

    if df is None:
        st.info(
            "No dataset found. Place a CSV at `data/raw/bugs.csv` or generate a "
            "sample with `python -m src.data_loader --make-sample`."
        )
        return

    # Ensure a `text` column exists for downstream charts
    if "text" not in df.columns:
        df["text"] = (
            df.get("summary", "").fillna("").astype(str)
            + " "
            + df.get("description", "").fillna("").astype(str)
        ).str.strip()

    st.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Class distribution", "Top words", "Text length", "Component × Severity"]
    )

    with tab1:
        st.plotly_chart(severity_pie(df), use_container_width=True)

    with tab2:
        top = top_words_per_class(df, clean_col="clean_text", top_k=10)
        classes_present = [c for c in SEVERITY_ORDER if c in top]
        if not classes_present:
            st.info("No data available for top-words view.")
        else:
            selected = st.selectbox("Severity class", classes_present)
            pairs = top.get(selected, [])
            if pairs:
                wdf = pd.DataFrame(pairs, columns=["Token", "Count"])
                fig = px.bar(
                    wdf,
                    x="Count",
                    y="Token",
                    orientation="h",
                    color_discrete_sequence=[SEVERITY_COLORS.get(selected, "#1e88e5")],
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Note: if `clean_text` column is absent, counts come from raw text "
                    "and will include stopwords."
                )

    with tab3:
        st.plotly_chart(text_length_box(df), use_container_width=True)

    with tab4:
        if "component" not in df.columns:
            st.info("The `component` column is not present in this dataset.")
        else:
            st.plotly_chart(component_severity_heatmap(df), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

def render_performance() -> None:
    st.title("📈 Model Performance")

    metrics = load_metrics()
    if not metrics:
        st.info("Run training to generate metrics.")
        return

    labels: list[str] = metrics.get("labels", [])
    results = metrics.get("results", {})

    st.markdown("### Metrics comparison")
    scalar_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    table = pd.DataFrame(
        {
            model: {k: results[model].get(k) for k in scalar_keys}
            for model in results
        }
    ).round(4)
    st.dataframe(table, use_container_width=True)

    # Confusion matrices + ROC need the test split + loaded models
    predictor = load_predictor()
    test_df = load_test_split()

    if test_df is None or test_df.empty:
        st.info("Test split (`models/test_split.csv`) not found — retrain to generate it.")
        return

    # CSV round-trip can turn whitespace-only rows back into NaN; replace
    # them with empty strings so the vectorizer accepts every row.
    test_df = test_df.dropna(subset=["severity_encoded"]).copy()
    test_df["clean_text"] = test_df["clean_text"].fillna("").astype(str)
    X = predictor.vectorizer.transform(test_df["clean_text"].to_numpy(dtype=object))
    y_true = test_df["severity_encoded"].astype(int).to_numpy()

    st.markdown("### Confusion matrices")
    cols = st.columns(len(predictor.available_models))
    for col, model_name in zip(cols, predictor.available_models):
        model = predictor.models[model_name]
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        with col:
            st.plotly_chart(
                confusion_heatmap(cm.tolist(), labels, title=model_name),
                use_container_width=True,
            )

    # Per-class F1 bar chart
    st.markdown("### Per-class F1 score")
    per_class_rows: list[dict[str, object]] = []
    for model_name, model_metrics in results.items():
        pcf1 = model_metrics.get("per_class_f1", {})
        for cls, val in pcf1.items():
            per_class_rows.append({"Model": model_name, "Class": cls, "F1": val})
    if per_class_rows:
        pcdf = pd.DataFrame(per_class_rows)
        ordered = [c for c in SEVERITY_ORDER if c in pcdf["Class"].unique()]
        fig = px.bar(
            pcdf,
            x="Class",
            y="F1",
            color="Model",
            barmode="group",
            category_orders={"Class": ordered},
        )
        fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    # ROC curves (one-vs-rest) for each model
    st.markdown("### ROC curves (one-vs-rest)")
    y_bin = label_binarize(y_true, classes=list(range(len(labels))))
    for model_name in predictor.available_models:
        model = predictor.models[model_name]
        if not hasattr(model, "predict_proba"):
            continue
        y_score = model.predict_proba(X)
        fig = go.Figure()
        for i, lbl in enumerate(labels):
            if y_bin.ndim == 1 or y_bin.shape[1] <= i:
                continue
            if np.sum(y_bin[:, i]) == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{lbl} (AUC={roc_auc:.2f})",
                    line=dict(color=SEVERITY_COLORS.get(lbl, "#607d8b")),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )
        fig.update_layout(
            title=model_name,
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Random Forest feature importance
    if "Random Forest" in predictor.models:
        st.markdown("### Random Forest — top feature importances")
        rf = predictor.models["Random Forest"]
        if hasattr(rf, "feature_importances_"):
            feature_names = np.array(predictor.vectorizer.get_feature_names_out())
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[-25:][::-1]
            fdf = pd.DataFrame(
                {
                    "Feature": feature_names[top_idx],
                    "Importance": importances[top_idx],
                }
            )
            fig = px.bar(
                fdf,
                x="Importance",
                y="Feature",
                orientation="h",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                height=600,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "home":
    render_home()
elif page == "predict":
    render_predict()
elif page == "insights":
    render_insights()
elif page == "performance":
    render_performance()
