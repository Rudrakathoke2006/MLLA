"""00_merge_datasets.py - merge up to 4 Kaggle loan datasets into ONE file (data/loan_data.csv).

Put the downloaded CSV files in data/raw/ with these names (missing ones are skipped):
    loan_prediction.csv   <- Kaggle: altruistdelhite04/loan-prediction-problem-dataset
    home_loan.csv         <- Kaggle: rishikeshkonapure/home-loan-approval
    loan_approval.csv     <- Kaggle: architsharma01/loan-approval-prediction-dataset
    credit_risk.csv       <- Kaggle: laotse/credit-risk-dataset
Every dataset is converted to the column format of dataset 1, so steps 01-06 work unchanged.
"""
import os
import sys
import numpy as np
import pandas as pd

RAW = "data/raw"
OUT = "data/loan_data.csv"
MAX_ROWS = 700      # cap per source so one big dataset cannot swamp the others
GOOD_CIBIL = 650     # assumption: CIBIL score >= 650 means Credit_History = 1
SEED = 42
COLS = ["Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area", "Loan_Status"]


def read(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]                       # some files have ' column' names
    for c in df.columns:                                               # and ' Graduate' style values
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    print(f"[ok]   {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def need(df, cols, tag):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] {tag}: columns {missing} not found -> dataset skipped.")
        print(f"       columns in this file: {list(df.columns)}")
        return False
    return True


def dep_text(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        return v
    return "3+" if v >= 3 else str(int(v))


def from_dataset1_format(df, tag):
    """Datasets 1 and 2 already use the Loan_ID / Gender / ... / Loan_Status format."""
    if not need(df, COLS[1:], tag):
        return None
    out = df.copy()
    if "Loan_ID" not in out:
        out["Loan_ID"] = ""
    out = out[COLS].copy()
    out["Dependents"] = out["Dependents"].map(dep_text)
    out["Loan_Status"] = out["Loan_Status"].map(
        lambda v: "Y" if str(v).strip().upper() in ("Y", "1", "APPROVED") else "N")
    return out


def from_loan_approval(df):
    """Dataset 3: income_annum, loan_amount, loan_term (years), cibil_score, loan_status."""
    cols = ["no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
            "loan_term", "cibil_score", "loan_status"]
    if not need(df, cols, "loan_approval"):
        return None
    out = pd.DataFrame(index=df.index)
    out["Loan_ID"] = ""
    out["Gender"] = np.nan
    out["Married"] = np.nan
    out["Dependents"] = df["no_of_dependents"].map(dep_text)
    out["Education"] = df["education"]
    out["Self_Employed"] = df["self_employed"]
    out["ApplicantIncome"] = df["income_annum"] / 12            # annual -> monthly
    out["CoapplicantIncome"] = 0.0
    out["LoanAmount"] = df["loan_amount"] / 1000                # -> thousands
    out["Loan_Amount_Term"] = df["loan_term"] * 12              # years -> months
    out["Credit_History"] = (df["cibil_score"] >= GOOD_CIBIL).astype(float)
    out["Property_Area"] = np.nan
    out["Loan_Status"] = np.where(df["loan_status"].astype(str).str.lower() == "approved", "Y", "N")
    return out[COLS]


def from_credit_risk(df):
    """Dataset 4: person_income, loan_amnt, cb_person_default_on_file, loan_status (1 = DEFAULT)."""
    cols = ["person_income", "loan_amnt", "cb_person_default_on_file", "loan_status"]
    if not need(df, cols, "credit_risk"):
        return None
    out = pd.DataFrame(index=df.index)
    out["Loan_ID"] = ""
    for c in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        out[c] = np.nan
    out["ApplicantIncome"] = df["person_income"] / 12
    out["CoapplicantIncome"] = 0.0
    out["LoanAmount"] = df["loan_amnt"] / 1000
    out["Loan_Amount_Term"] = np.nan
    out["Credit_History"] = (df["cb_person_default_on_file"] == "N").astype(float)
    out["Loan_Status"] = np.where(df["loan_status"] == 0, "Y", "N")   # default -> not eligible
    return out[COLS]


def rescale_money(part, ref_median, tag):
    """Different datasets use different currencies/scales. Multiply income and loan amount by ONE
    factor so the median income matches the reference. Ratios (EMI/income) stay unchanged."""
    m = part["ApplicantIncome"].median()
    if pd.notna(m) and m > 0:
        f = ref_median / m
        for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
            part[c] = part[c] * f
        print(f"       {tag}: money columns rescaled by factor {f:.4f}")
    return part


sources = [("loan_prediction.csv", "loan_prediction", from_dataset1_format),
           ("home_loan.csv", "home_loan", from_dataset1_format),
           ("loan_approval.csv", "loan_approval", lambda d, t: from_loan_approval(d)),
           ("credit_risk.csv", "credit_risk", lambda d, t: from_credit_risk(d))]

parts, ref_median = [], None
for fname, tag, fn in sources:
    df = read(fname)
    if df is None:
        continue
    part = fn(df, tag)
    if part is None:
        continue
    if ref_median is None:                         # first usable dataset defines the money scale
        ref_median = part["ApplicantIncome"].median()
    elif fn is not from_dataset1_format:           # datasets 1 and 2 share units: never rescale them
        part = rescale_money(part, ref_median, tag)
    if len(part) > MAX_ROWS:
        part = part.sample(MAX_ROWS, random_state=SEED)
        print(f"       {tag}: randomly sampled {MAX_ROWS} rows")
    part["_source"] = tag
    parts.append(part)

if not parts:
    sys.exit("No usable dataset found in data/raw/. Check the file names and the columns.")

merged = pd.concat(parts, ignore_index=True)
before = len(merged)
merged = merged.drop_duplicates(subset=COLS[1:])   # same applicant in 2 files
print(f"Removed {before - len(merged)} duplicate applicants (e.g. dataset 2 is often a copy of dataset 1)")
merged = merged.sample(frac=1, random_state=SEED).reset_index(drop=True)
merged["Loan_ID"] = [f"M{i:06d}" for i in range(len(merged))]

print("\nRows and approval rate per source:")
print(merged.groupby("_source")["Loan_Status"].agg(rows="count", approved=lambda s: (s == "Y").mean()).round(3))
merged = merged.drop(columns="_source")
merged.to_csv(OUT, index=False)
print(f"\nSaved {OUT}: {merged.shape[0]} rows, {merged.shape[1]} columns")
print("Missing values per column:\n", merged.isnull().sum()[merged.isnull().sum() > 0].to_string())
