# 🐞 Bug Severity Classifier

A machine-learning system that classifies bug reports into seven severity
levels — **Blocker, Critical, Major, Normal, Minor, Trivial, Enhancement** —
using TF-IDF features with Logistic Regression and Random Forest, plus an
interactive Streamlit dashboard for predictions and dataset insights.

## Objectives

- Automate the initial triage step of assigning severity to incoming bug
  reports, which is typically done manually by maintainers.
- Compare two classical ML approaches (linear + ensemble) on the same features
  to understand the accuracy / interpretability trade-off.
- Provide an interpretable dashboard so reviewers can see not just *what* the
  model predicts but *why*.

## Project structure

```
bug-severity-classifier/
├── data/
│   ├── raw/                    # Original Bugzilla/Eclipse CSV
│   └── processed/              # Cleaned data
├── models/                     # Saved .pkl + metrics + confusion matrices
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_loader.py          # Load & clean bug data + synthetic sample
│   ├── preprocessor.py         # Text cleaning, tokenization, stopwords
│   ├── feature_engineering.py  # TF-IDF vectorizer
│   ├── train.py                # Train LogReg + Random Forest
│   ├── evaluate.py             # Metrics, confusion matrix
│   └── predict.py              # Inference pipeline
├── app/
│   ├── streamlit_app.py        # Multi-page dashboard
│   └── utils.py                # UI helpers
├── tests/
│   └── test_preprocessor.py
├── requirements.txt
└── README.md
```

## Dataset

The pipeline is built for Eclipse/Bugzilla-style CSV exports with these
columns:

| Column        | Description                                       |
| ------------- | ------------------------------------------------- |
| `bug_id`      | Unique bug identifier                             |
| `summary`     | One-line title of the bug                         |
| `description` | Longer free-text description                      |
| `severity`    | One of the 7 classes above                        |
| `component`   | Subsystem / module                                |
| `priority`    | P1..P5                                            |

Place your CSV at `data/raw/bugs.csv`.

### Getting real data

Eclipse bug reports are available from the Eclipse Bugzilla REST API — e.g.
`https://bugs.eclipse.org/bugs/rest/bug?product=Platform&limit=1000`. Older
research datasets (e.g. the MSR Mining Challenge Eclipse dataset) can also
be used. Export/convert to a CSV with the columns listed above.

### No data yet? Use the sample

```bash
python -m src.data_loader --make-sample
```

This writes a small synthetic `data/raw/sample.csv` that covers all 7 classes
— enough to verify the pipeline end-to-end.

### Using the MSR Eclipse JSON dump

If your `data/raw/` contains the MSR-style per-column JSON files
(`severity.json`, `short_desc.json`, `component.json`, `priority.json`, ...),
convert them into a single CSV first:

```bash
python -m src.msr_adapter --raw-dir data/raw --out data/raw/bugs.csv
```

The adapter extracts the **latest** value from each bug's change history.
The MSR dump has no full description, so `description` is left empty and
the classifier trains on the short summary only.

## Preprocessing pipeline

1. Lowercase + whitespace normalisation
2. Strip URLs, email addresses, file paths, stack-trace frames
3. Remove special characters (keep alphanumerics + basic punctuation)
4. Tokenize (NLTK `punkt`)
5. Drop NLTK English stopwords + custom tech stopwords (`bug`, `issue`, …)
6. Lemmatize with WordNet
7. Re-join into a space-separated string

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First run only) NLTK resources will be downloaded on demand.
#    If you are on a machine without internet, pre-download with:
#    python -c "import nltk; [nltk.download(x) for x in ('stopwords','wordnet','punkt','punkt_tab')]"
```

## Train

```bash
# Use your own dataset
python -m src.train --data data/raw/bugs.csv

# With SMOTE for class imbalance
python -m src.train --data data/raw/bugs.csv --smote

# Smoke-test the pipeline on synthetic data
python -m src.train --make-sample
```

After training, `models/` will contain:

- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `logreg_model.pkl`, `rf_model.pkl`
- `metrics.json`, `test_split.csv`
- `confusion_matrix_logreg.png`, `confusion_matrix_rf.png`

## Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Dashboard pages

- **🏠 Home** — dataset stats + accuracy / F1 metric cards for both models.
- **🔮 Predict Severity** — enter a bug summary/description, pick a model,
  see the predicted class (color-coded badge), confidence bar, probability
  distribution, and (for Logistic Regression) the top 10 influential tokens.
- **📊 Data Insights** — severity pie chart, top tokens per class, text-length
  box plot, component × severity heatmap.
- **📈 Model Performance** — metrics comparison table, interactive confusion
  matrices, per-class F1 bars, one-vs-rest ROC curves, and Random Forest
  feature importance.

## Tests

```bash
pytest tests/ -v
```

## Screenshots

Place screenshots of the dashboard pages under `docs/screenshots/` and link
them here:

- _Home page_ — `docs/screenshots/home.png`
- _Predict page_ — `docs/screenshots/predict.png`
- _Data Insights_ — `docs/screenshots/insights.png`
- _Model Performance_ — `docs/screenshots/performance.png`

## Results summary

Metrics are written to `models/metrics.json` after each training run. A
typical run on the synthetic sample looks like this (your real-data results
will differ):

| Model               | Accuracy | F1 (macro) | F1 (weighted) |
| ------------------- | -------- | ---------- | ------------- |
| Logistic Regression | *varies* | *varies*   | *varies*      |
| Random Forest       | *varies* | *varies*   | *varies*      |

## Future improvements

- Try transformer-based embeddings (BERT / DistilBERT) for the same task.
- Add SHAP explanations on the Predict page.
- Fine-tune the severity class set (some projects merge Trivial → Minor).
- Support streaming / batch inference from Bugzilla's REST API.
- Add CI (lint + tests) and Docker packaging.

## License

MIT.
