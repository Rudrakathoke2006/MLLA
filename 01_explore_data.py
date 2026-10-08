"""01_explore_data.py - look at the data before doing anything else."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports", exist_ok=True)
df = pd.read_csv("data/loan_data.csv")

print("Shape (rows, columns):", df.shape)
print(df.head())
df.info()
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df["Loan_Status"].value_counts())
print(df["Loan_Status"].value_counts(normalize=True).round(3))

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.countplot(x="Loan_Status", data=df, ax=ax[0, 0])
sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=ax[0, 1])
sns.countplot(x="Property_Area", hue="Loan_Status", data=df, ax=ax[1, 0])
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df, ax=ax[1, 1])
plt.tight_layout()
plt.savefig("reports/01_overview.png", dpi=120)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.savefig("reports/01_correlation.png", dpi=120, bbox_inches="tight")
plt.close()
print("Charts saved in reports/")