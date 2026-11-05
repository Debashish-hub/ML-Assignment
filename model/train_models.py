"""Train, tune and evaluate five classifiers on the IBM Telco Customer Churn dataset.

Run from the project root:
    python -m model.train_models

Outputs
    model/<name>.joblib    full sklearn Pipeline (preprocessing + classifier)
    model/metrics.csv      6 evaluation metrics per model on the held-out test set
    model/best_params.json hyper-parameters chosen by cross-validation
    test_data.csv          held-out 20% split (raw features + target) for the app
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.data_prep import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                             TARGET_COL, clean_frame, encode_target,
                             positive_scores)

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "model"
SEED = 322          # change this to your own number before submitting
TEST_SIZE = 0.20


def build_preprocessor() -> ColumnTransformer:
    # Dense one-hot output because GaussianNB cannot consume sparse matrices.
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


# name -> (file stem, estimator, small hyper-parameter grid)
MODELS = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [3, 5, 7, 10, None], "clf__min_samples_leaf": [1, 10, 30],
         "clf__class_weight": [None, "balanced"]},
    ),
    "KNN": (
        "knn",
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21, 31, 41], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        "naive_bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    ),
    "Support Vector Machine": (
        "svm",
        SVC(random_state=SEED),   # AUC uses decision_function scores
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"],
         "clf__class_weight": [None, "balanced"]},
    ),
}


def evaluate(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    df = clean_frame(pd.read_csv(RAW_DATA))
    df[TARGET_COL] = encode_target(df[TARGET_COL])
    print(f"Dataset: {df.shape[0]} rows x {len(FEATURES)} features | churn rate = {df[TARGET_COL].mean():.3f}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=SEED)

    # Save the untouched test split (Yes/No labels, like the raw data) for the app.
    test_out = test_df[FEATURES + [TARGET_COL]].copy()
    test_out[TARGET_COL] = test_out[TARGET_COL].map({1: "Yes", 0: "No"})
    test_out.to_csv(ROOT / "test_data.csv", index=False)

    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURES], test_df[TARGET_COL]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, best_params = [], {}
    for name, (stem, estimator, grid) in MODELS.items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        # F1 as the selection metric: churners are the minority (~26%), so
        # accuracy alone would reward a model that rarely predicts churn.
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_prob = positive_scores(best, X_test)
        metrics = evaluate(y_test, y_pred, y_prob)
        rows.append({"Model": name, **metrics})
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        joblib.dump(best, MODEL_DIR / f"{stem}.joblib")
        print(f"{name:<24} CV-F1={search.best_score_:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
              + f"  ({time.time() - t0:.1f}s)")

    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(MODEL_DIR / "metrics.csv")
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    print("\n" + results.to_string())


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
