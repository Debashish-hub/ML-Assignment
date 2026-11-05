"""Shared data-preparation helpers used by both the training script and the Streamlit app.

Keeping cleaning logic in one place guarantees that the app transforms uploaded
test data exactly the way the training data was transformed.
"""

import pandas as pd

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 19 input features

CLASS_LABELS = ["No Churn", "Churn"]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix raw-data quirks of the Telco dataset.

    * TotalCharges is stored as text and is blank for brand-new customers
      (tenure = 0). A blank there genuinely means "nothing billed yet", so 0 is
      the correct value rather than a mean/median guess.
    * The customer ID carries no predictive signal and is dropped.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def encode_target(y: pd.Series) -> pd.Series:
    """Map Yes/No (or already-numeric 1/0) churn labels to 1/0."""
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    mapped = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    return mapped


def positive_scores(model, X):
    """Continuous score for the positive class, used for AUC / ROC.

    SVC is trained without probability=True (deprecated in recent sklearn and
    5x slower), so we fall back to its signed distance from the margin. AUC only
    depends on the ranking of scores, so this is exactly as valid as a probability.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def missing_columns(df: pd.DataFrame, need_target: bool = True) -> list[str]:
    required = FEATURES + ([TARGET_COL] if need_target else [])
    return [c for c in required if c not in df.columns]
