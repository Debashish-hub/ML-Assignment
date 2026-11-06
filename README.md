# ChurnLens — Telecom Customer Churn Classification

**M.Tech (AIML), BITS Pilani WILP — Machine Learning, Assignment 2**

🔗 **GitHub repository:** https://github.com/Debashish-hub/ML-Assignment
🔗 **Live Streamlit app:** https://ml-assignment-edhzemsmar3c8bcggop5er.streamlit.app/

---

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels ("churns"). Retaining an
existing customer is far cheaper than acquiring a new one, so the business wants to flag
customers who are **likely to churn** early enough for the retention team to act.

This project frames that as a **binary classification** problem — predict `Churn` (Yes / No)
from a customer's demographics, subscribed services and billing details — and compares five
classical ML classifiers on the same held-out test set. The trained models are served through
an interactive Streamlit app that scores any uploaded test CSV without retraining.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | IBM Telco Customer Churn |
| Source | IBM sample data (also published on Kaggle as *Telco Customer Churn* by BlastChar) |
| Instances | **7,043** customers |
| Input features | **19** (4 numeric + 15 categorical); `customerID` dropped as an identifier |
| Target | `Churn` — Yes (1,869 · 26.5%) / No (5,174 · 73.5%) → moderately imbalanced |
| Train / test split | 80 / 20, stratified on `Churn` → 5,634 train, 1,409 test (`test_data.csv`) |

**Features**

| Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone | `PhoneService`, `MultipleLines` |
| Internet & add-ons | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Preprocessing**

- `TotalCharges` is stored as text and is blank for 11 customers with `tenure = 0`; these are
  new customers who have not been billed yet, so blanks are set to **0** (not mean-imputed).
- Numeric features → `StandardScaler` (essential for KNN, SVM and Logistic Regression, which
  are distance- or margin-based).
- Categorical features → `OneHotEncoder(handle_unknown="ignore")`.
- Preprocessing and classifier are saved together as one sklearn `Pipeline` per model, so the
  app applies exactly the same transformations to uploaded data.

## c. GitHub repository

`<paste your repo URL here>`

```
project-folder/
├── app.py                    # Streamlit app (loads saved models, never retrains)
├── requirements.txt
├── README.md
├── test_data.csv             # held-out 20% test split used in all experiments
├── data/
│   └── telco_churn.csv       # raw dataset
├── model/
│   ├── train_models.py       # training, tuning, evaluation script
│   ├── data_prep.py          # shared cleaning / feature lists
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics.csv           # metrics table below
│   └── best_params.json      # hyper-parameters chosen by CV
└── .streamlit/config.toml    # app theme
```

**Reproduce:** `pip install -r requirements.txt` → `python -m model.train_models` → `streamlit run app.py`

## d. Models used

All five models were tuned with **5-fold stratified cross-validation (GridSearchCV) on the
training split only**, using **F1** as the selection score (accuracy would reward a model that
rarely predicts the minority churn class). Metrics below are on the untouched test split.

| Model | Key hyper-parameters selected |
|---|---|
| Logistic Regression | C = 10, class_weight = balanced |
| Decision Tree | max_depth = 3, class_weight = balanced |
| KNN | k = 31, uniform weights |
| Naive Bayes (Gaussian) | var_smoothing = 1e-9 |
| SVM | RBF kernel, C = 1, class_weight = balanced |

### Comparison table (test set, n = 1,409)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7452 | **0.8366** | 0.5129 | 0.7968 | **0.6241** | 0.4695 |
| Decision Tree | 0.7374 | 0.8065 | 0.5035 | 0.7701 | 0.6089 | 0.4457 |
| KNN | **0.8006** | 0.8258 | **0.6317** | 0.5963 | 0.6135 | **0.4796** |
| Naive Bayes | 0.6948 | 0.8214 | 0.4588 | **0.8342** | 0.5920 | 0.4230 |
| Support Vector Machine | 0.7331 | 0.8163 | 0.4983 | 0.7781 | 0.6075 | 0.4437 |

*AUC for SVM is computed from `decision_function` scores (AUC depends only on the ranking of
scores, so probabilities are not required).*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.837) and F1 (0.624). With `class_weight="balanced"` it trades some precision for high recall (0.80), catching 4 in 5 churners. Its strong showing indicates churn risk is largely a smooth, near-linear function of a few drivers (short tenure, month-to-month contract, fibre internet, electronic-cheque payment). Coefficients are also interpretable for the business. |
| Decision Tree | CV picked a very shallow tree (depth 3) — deeper trees overfit. It is the most explainable model (a few if-then rules: contract type first, then internet type and tenure) but has the lowest AUC (0.807) because a depth-3 tree outputs only a handful of distinct probability levels, giving a coarse ranking. |
| KNN | Best accuracy (0.801), precision (0.632) and MCC (0.480), but the lowest recall (0.596): without class weighting it leans toward the majority "No" class and misses ~40% of churners. Needed a large k (31) to smooth noise across 45 scaled + one-hot dimensions, and it is the slowest to predict because it stores the full training set. |
| Naive Bayes | Highest recall (0.834) but lowest precision (0.459), accuracy and MCC — it over-predicts churn. Gaussian NB assumes independent, normally distributed features, which one-hot and strongly correlated features (`tenure`, `TotalCharges`, `MonthlyCharges`) violate. Still a respectable AUC (0.821) — its ranking is good even if its calibrated threshold is not. |
| Support Vector Machine | Balanced RBF-SVM behaves much like Logistic Regression (recall 0.78, F1 0.608) but slightly worse on every metric and far costlier to tune. The non-linear kernel adds no value here, again pointing to a mostly linear decision boundary. |
| **Overall winner for this dataset** | **Logistic Regression.** It has the best AUC and F1 and the second-best MCC, while catching ~80% of churners — the outcome a retention team cares about most. KNN wins on accuracy/MCC but misses too many churners. Logistic Regression is also the fastest and most interpretable model, so it is the best practical choice. |


## How to run locally

```bash
git clone https://github.com/Debashish-hub/ML-Assignment.git
cd ML-Assignment
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m model.train_models      # retrains and regenerates model/ and test_data.csv
.venv/bin/python -m streamlit run app.py
```

Requires Python 3.11+. On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Streamlit app features

- **CSV upload** (test data only) plus a one-click sample `test_data.csv` download / bundled option
- **Model selection dropdown** for the five classifiers
- **All six evaluation metrics** for the selected model
- **Confusion matrix** heatmap with plain-language TP/FN/FP/TN breakdown
- **Classification report**, **ROC curve**, side-by-side **comparison of all models** on the uploaded data, and downloadable **predictions**
- Validates the upload (missing columns, invalid labels) with clear error messages
