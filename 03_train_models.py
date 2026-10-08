"""03_train_models.py - tune 7 models with cross-validation and save them."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from features import CAT_COLS, NUM_COLS

X_train = pd.read_csv("data/X_train.csv", dtype={"Dependents": str})
y_train = pd.read_csv("data/y_train.csv")["Loan_Status"]

numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_steps, NUM_COLS),
    ("cat", categorical_steps, CAT_COLS),
])

models = {
    "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "Naive Bayes": (GaussianNB(),
                    {"clf__var_smoothing": np.logspace(-9, -2, 8)}),
    "SVM": (SVC(probability=True, class_weight="balanced", random_state=42),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"],
             "clf__gamma": ["scale", 0.01, 0.1]}),
    "KNN": (KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                      {"clf__max_depth": [3, 4, 5, 7, 10], "clf__min_samples_split": [2, 5, 10],
                       "clf__criterion": ["gini", "entropy"]}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=42),
                      {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [4, 6, 8, None],
                       "clf__min_samples_split": [2, 5]}),
    "AdaBoost": (AdaBoostClassifier(random_state=42),
                 {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 0.5, 1.0]}),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuned = {}
for name, (model, grid) in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    print(f"{name:20s} best CV ROC-AUC = {search.best_score_:.3f}")
    print("   best settings:", search.best_params_)

joblib.dump(tuned, "models/tuned_models.joblib")
print("Saved models/tuned_models.joblib")