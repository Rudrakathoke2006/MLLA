"""06_extra_models.py - add boosting models and compare everything on the SAME split."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import CAT_COLS, NUM_COLS

X_train = pd.read_csv("data/X_train.csv", dtype={"Dependents": str})
X_test = pd.read_csv("data/X_test.csv", dtype={"Dependents": str})
y_train = pd.read_csv("data/y_train.csv")["Loan_Status"]
y_test = pd.read_csv("data/y_test.csv")["Loan_Status"]

prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLS),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
])

extra = {
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.03, 0.1],
                           "clf__max_depth": [2, 3]}),
    "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=42),
                             {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3, None],
                              "clf__max_iter": [50, 100]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
                {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.03, 0.1],
                 "clf__max_depth": [2, 3, 4]}),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, tuned_extra = [], {}
for name, (model, grid) in extra.items():
    pipe = Pipeline([("prep", prep), ("clf", model)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
    proba = search.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, proba),
                 "CV ROC-AUC": search.best_score_})
    tuned_extra[name] = search.best_estimator_
    print(name, "best settings:", search.best_params_)

base = pd.read_csv("models/metrics.csv")
combined = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
combined = combined.sort_values(["ROC-AUC", "F1"], ascending=False).reset_index(drop=True)
combined.to_csv("models/metrics_extended.csv", index=False)
joblib.dump(tuned_extra, "models/tuned_extra_models.joblib")
print(combined.round(3).to_string())