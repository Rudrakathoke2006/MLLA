"""features.py - shared helpers used by the training scripts AND the web app."""
import numpy as np
import pandas as pd

CAT_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUM_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "Total_Income", "EMI", "Income_to_Loan", "Debt_to_Income", "Log_Total_Income", "Log_LoanAmount",
]
FEATURES = CAT_COLS + NUM_COLS


def add_features(df):
    df = df.copy()
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / term
    df["Income_to_Loan"] = df["Total_Income"] / (df["LoanAmount"] * 1000 + 1)
    df["Debt_to_Income"] = (df["EMI"] * 1000) / (df["Total_Income"] + 1)
    df["Log_Total_Income"] = np.log1p(df["Total_Income"])
    df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])
    return df