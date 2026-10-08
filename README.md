# MLLA: Machine Learning for Loan Approval & Credit Risk Intelligence 🏦

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB3324.svg)](https://xgboost.readthedocs.io/)

**LendIQ / MLLA** is an end-to-end, production-grade machine learning system designed to automate loan underwriting, evaluate applicant credit risk, and provide transparent model consensus. 

The project benchmarks **10 machine learning algorithms** (classical models alongside state-of-the-art boosting architectures), calibrates probability predictions using Platt scaling, chooses an optimal decision threshold, and serves the system via an interactive, light-themed Streamlit application.

---

## 🚀 Key Features

- **Multi-Algorithm Benchmark (10 Models):** Evaluated AdaBoost, Logistic Regression, Random Forest, Gradient Boosting, XGBoost, HistGradientBoosting, Naive Bayes, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Decision Trees.
- **Domain Feature Engineering:** Engineered vital financial ratios including:
  - `Total_Income = ApplicantIncome + CoapplicantIncome`
  - `EMI = LoanAmount / Loan_Amount_Term`
  - `Debt_to_Income = (EMI * 1000) / (Total_Income + 1)`
  - `Income_to_Loan = Total_Income / (LoanAmount * 1000 + 1)`
  - Outlier stabilization via `np.log1p` on income and loan amounts.
- **Probability Calibration & Threshold Tuning:** Employed Sigmoid (`CalibratedClassifierCV`) calibration to eliminate tree-ensemble probability distortion, and tuned the decision threshold to `0.49` via 5-Fold Stratified Cross-Validation on out-of-fold training data.
- **Interactive Web App (Streamlit):**
  - Modern light theme with responsive cards, badges, and progress meters.
  - One-click personas: Prime Salaried, High DTI Borderline, and Poor Credit History.
  - Live financial calculators (EMI and DTI %) before submission.
  - Multi-model consensus engine showing agreement across all 10 algorithms.
  - Dedicated tabs for Underwriting, Model Benchmarks, EDA, and Pipeline Architecture.

---

## 📊 Benchmark Results (Hold-out Test Set)

Evaluated on 123 unseen applicants (20% hold-out test set):

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **AdaBoost** *(Best Calibrated)* | **87.8%** | **88.0%** | **95.3%** | **0.915** | **0.894** |
| 🥈 | **Logistic Regression** | 84.6% | 87.5% | 90.6% | 0.890 | 0.872 |
| 🥉 | **Random Forest** | 82.1% | 90.9% | 82.4% | 0.864 | 0.867 |
| 4 | **Gradient Boosting** | 86.2% | 85.4% | 96.5% | 0.906 | 0.862 |
| 5 | **XGBoost** | 83.7% | 84.9% | 92.9% | 0.888 | 0.855 |
| 6 | **HistGradientBoosting** | **89.4%** | 89.1% | 96.5% | **0.927** | 0.854 |
| 7 | **Naive Bayes** | 83.7% | 83.5% | 95.3% | 0.890 | 0.840 |
| 8 | **SVM** | 82.9% | 84.0% | 92.9% | 0.883 | 0.805 |
| 9 | **KNN** | 84.6% | 83.7% | 96.5% | 0.896 | 0.803 |
| 10 | **Decision Tree** | 81.3% | 83.0% | 91.8% | 0.872 | 0.776 |

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml          # Light theme styling and server configuration
├── data/
│   ├── loan_data.csv        # Raw dataset (614 rows, 13 features)
│   ├── X_train.csv          # Pre-split training features (491 rows)
│   ├── X_test.csv           # Hold-out test features (123 rows)
│   ├── y_train.csv          # Training labels
│   └── y_test.csv           # Test labels
├── models/
│   ├── tuned_models.joblib        # 7 base tuned models
│   ├── tuned_extra_models.joblib  # 3 boosting models (Gradient Boosting, Hist, XGBoost)
│   ├── final_model.joblib         # Calibrated production classifier
│   ├── final_settings.json        # Optimal threshold & best model metadata
│   ├── metrics.csv                # Baseline performance metrics
│   ├── metrics_extended.csv       # Extended 10-model benchmark comparison
│   └── feature_importance.csv     # Permutation feature importances
├── reports/
│   ├── 01_overview.png            # EDA distributions & count plots
│   ├── 01_correlation.png         # Correlation heatmap
│   ├── 04_roc_curves.png          # Test set ROC comparison curves
│   └── 04_confusion_matrix.png    # Confusion matrix of champion model
├── 01_explore_data.py       # Exploratory Data Analysis & visual reporting
├── 02_prepare_data.py       # Cleaning, feature engineering, and stratified train/test split
├── 03_train_models.py       # 5-fold CV hyperparameter tuning of 7 base models
├── 04_evaluate_models.py    # Hold-out evaluation, metrics calculation, and ROC generation
├── 05_finalize_model.py     # Probability calibration, F1 threshold tuning, and feature importance
├── 06_extra_models.py       # SOTA boosting models (Gradient Boosting, Hist, XGBoost)
├── features.py              # Centralized domain feature engineering helper
├── risk.py                  # Underwriting decision rules & risk-tier logic
├── app.py                   # Streamlit interactive UI application
├── requirements.txt         # Pinned project dependencies
└── .gitignore               # Excludes venv, cache, and OS artifacts
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Rudrakathoke2006/MLLA.git
cd MLLA
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Mac / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Pipeline (Optional - Pre-trained Artifacts Included)
To retrain and regenerate all models and metrics from scratch:
```bash
python 01_explore_data.py
python 02_prepare_data.py
python 03_train_models.py
python 04_evaluate_models.py
python 05_finalize_model.py
python 06_extra_models.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
