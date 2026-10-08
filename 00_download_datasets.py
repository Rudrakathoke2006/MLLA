"""00_download_datasets.py - download the 4 public loan datasets into data/raw/ (no Kaggle login needed)."""
import os
import urllib.request
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

DATASETS = {
    "loan_prediction.csv": "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv",
    "home_loan.csv": "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv",
    "loan_approval.csv": "https://raw.githubusercontent.com/cavinlobo/Loan-Approval-Prediction/main/loan_approval_dataset.csv",
    "credit_risk.csv": "https://raw.githubusercontent.com/JoshLG18/DSE-EMP-Project/main/Project/Data/credit_risk_dataset.csv",
}

for name, url in DATASETS.items():
    path = os.path.join("data", "raw", name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as f:
            f.write(resp.read())
        shape = pd.read_csv(path).shape
        print(f"[ok]   {name:20s} rows={shape[0]:6d} columns={shape[1]}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")