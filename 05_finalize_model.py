"""05_finalize_model.py - calibrate, choose a threshold, explain, and save the final model."""
import json
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

tuned = joblib.load("models/tuned_models.joblib")
best_name = open("models/best_name.txt").read().strip()
best_model = tuned[best_name]

X_train = pd.read_csv("data/X_train.csv", dtype={"Dependents": str})
X_test = pd.read_csv("data/X_test.csv", dtype={"Dependents": str})
y_train = pd.read_csv("data/y_train.csv")["Loan_Status"]
y_test = pd.read_csv("data/y_test.csv")["Loan_Status"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1) Calibration
calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=cv)
calibrated.fit(X_train, y_train)
raw_p = best_model.predict_proba(X_test)[:, 1]
cal_p = calibrated.predict_proba(X_test)[:, 1]
print("Brier score (lower is better)  raw:", round(brier_score_loss(y_test, raw_p), 4),
      " calibrated:", round(brier_score_loss(y_test, cal_p), 4))

# 2) Threshold from TRAINING data only
oof = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
candidates = [t / 100 for t in range(20, 81)]
scores = [f1_score(y_train, oof >= t) for t in candidates]
threshold = candidates[scores.index(max(scores))]
print("Chosen threshold:", threshold)

# 3) Final check on the test set
pred = (cal_p >= threshold).astype(int)
print("Final test  Accuracy:", round(accuracy_score(y_test, pred), 3),
      " F1:", round(f1_score(y_test, pred), 3), " ROC-AUC:", round(roc_auc_score(y_test, cal_p), 3))

# 4) Feature importance
imp = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                             n_repeats=15, random_state=42)
importance = (pd.DataFrame({"Feature": X_test.columns, "Importance": imp.importances_mean})
              .sort_values("Importance", ascending=False))
print(importance.head(8).round(3).to_string(index=False))

# 5) Save
joblib.dump(calibrated, "models/final_model.joblib")
settings = {"best_model": best_name, "threshold": threshold}
json.dump(settings, open("models/final_settings.json", "w"))
importance.to_csv("models/feature_importance.csv", index=False)
print("Saved final_model.joblib, final_settings.json, feature_importance.csv")