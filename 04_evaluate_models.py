"""04_evaluate_models.py - compare all models on the unseen test set."""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

tuned = joblib.load("models/tuned_models.joblib")
X_test = pd.read_csv("data/X_test.csv", dtype={"Dependents": str})
y_test = pd.read_csv("data/y_test.csv")["Loan_Status"]

rows, roc_data = [], {}
for name, model in tuned.items():
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
    roc_data[name] = roc_curve(y_test, proba)[:2]

results = pd.DataFrame(rows).sort_values(["ROC-AUC", "F1"], ascending=False).reset_index(drop=True)
results.to_csv("models/metrics.csv", index=False)
print(results.round(3).to_string())

best_name = results.loc[0, "Model"]
print("\nBest model:", best_name)
with open("models/best_name.txt", "w") as f:
    f.write(best_name)

plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_data.items():
    auc = results.loc[results["Model"] == name, "ROC-AUC"].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (test set)")
plt.legend(loc="lower right")
plt.savefig("reports/04_roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()

ConfusionMatrixDisplay.from_estimator(
    tuned[best_name], X_test, y_test,
    display_labels=["Rejected", "Approved"], cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.savefig("reports/04_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()
print("Plots saved in reports/")