"""02_prepare_data.py - clean the data, add features, split into train and test."""
import pandas as pd
from sklearn.model_selection import train_test_split
from features import add_features, FEATURES

df = pd.read_csv("data/loan_data.csv")
df = df.drop(columns=["Loan_ID"])
df = df.dropna(subset=["Loan_Status"])
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df = df.drop_duplicates()
df = add_features(df)

X = df[FEATURES]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Approval rate  train:", round(y_train.mean(), 3), " test:", round(y_test.mean(), 3))

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("Saved 4 files in data/")