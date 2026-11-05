"""ChurnLens — Streamlit front-end for the Telco churn classifiers.

Upload a labelled test CSV, pick a model, and the app scores it with the
pre-trained pipeline from model/ (no retraining happens here).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from model.data_prep import (CLASS_LABELS, FEATURES, TARGET_COL, clean_frame,
                             encode_target, missing_columns, positive_scores)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

ACCENT = "#0F766E"

st.set_page_config(page_title="ChurnLens", page_icon="📉", layout="wide")


# ---------- cached loaders ----------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models…")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def score_model(model, X, y):
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return y_pred, y_score, compute_metrics(y, y_pred, y_score)


# ---------- sidebar -------------------------------------------------------------
models = load_models()

with st.sidebar:
    st.markdown("### 1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + `Churn` column)", type=["csv"])
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv",
                           width="stretch")
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)

    st.markdown("### 2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))

    st.markdown("---")
    st.caption("Models were trained offline (`python -m model.train_models`). "
               "This app only loads them and predicts — it never retrains.")

# ---------- header ----------------------------------------------------------------
st.markdown(
    f"<h1 style='margin-bottom:0'>📉 ChurnLens</h1>"
    f"<p style='color:{ACCENT};font-size:1.05rem;margin-top:0.2rem'>"
    "Telecom customer-churn prediction · 5 classifiers compared on held-out data</p>",
    unsafe_allow_html=True,
)

# ---------- load data ----------------------------------------------------------
if uploaded is not None and not use_sample:
    raw = read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    raw = read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"
else:
    st.info("⬅️ Upload a test CSV in the sidebar, or tick **Use bundled test_data.csv**.")
    st.stop()

df = clean_frame(raw)
missing = missing_columns(df, need_target=True)
if missing:
    st.error(f"The uploaded file is missing required column(s): {', '.join(missing)}")
    st.stop()

y_true = encode_target(df[TARGET_COL])
if y_true.isna().any():
    st.error("`Churn` must contain only Yes/No (or 1/0) values.")
    st.stop()
y_true = y_true.astype(int)
X = df[FEATURES]

c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", f"{len(X):,}")
c2.metric("Input features", len(FEATURES))
c3.metric("Actual churn rate", f"{y_true.mean():.1%}")
st.caption(f"Source: **{source}**")

# ---------- selected model -----------------------------------------------------
model = models[model_name]
y_pred, y_score, metrics = score_model(model, X, y_true)

st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

tab_cm, tab_rep, tab_roc, tab_cmp, tab_pred = st.tabs(
    ["Confusion matrix", "Classification report", "ROC curve", "Compare all models", "Predictions"])

with tab_cm:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
                    annot_kws={"size": 13})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="content")
        plt.close(fig)
    with right:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"- **True positives (caught churners):** {tp}\n"
            f"- **False negatives (missed churners):** {fn}\n"
            f"- **False positives (false alarms):** {fp}\n"
            f"- **True negatives:** {tn}\n\n"
            "For a retention team, a missed churner (FN) is usually costlier than a "
            "false alarm (FP), so recall matters more than raw accuracy here."
        )

with tab_rep:
    rep = classification_report(y_true, y_pred, target_names=CLASS_LABELS,
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

with tab_roc:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {metrics['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    st.pyplot(fig, width="content")
    plt.close(fig)

with tab_cmp:
    rows = []
    for name, m in models.items():
        _, _, mt = score_model(m, X, y_true)
        rows.append({"Model": name, **mt})
    comp = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#CCFBF1"),
                 width="stretch")
    metric_pick = st.radio("Plot metric", list(comp.columns), index=5, horizontal=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    order = comp[metric_pick].sort_values()
    ax.barh(order.index, order.values,
            color=[ACCENT if n == model_name else "#94A3B8" for n in order.index])
    ax.set_xlabel(metric_pick)
    for i, v in enumerate(order.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with tab_pred:
    out = raw.copy()
    out["Predicted_Churn"] = np.where(y_pred == 1, "Yes", "No")
    out["Churn_Score"] = np.round(y_score, 4)
    st.dataframe(out.head(200), width="stretch")
    st.download_button("Download all predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
                       mime="text/csv")
