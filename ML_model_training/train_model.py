import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================================
# STEP 1: Load and Explore Dataset
# ==========================================================

print("BREAST CANCER CLASSIFICATION — ML TRAINING PIPELINE")
print("=" * 60)

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(f"\nDataset Shape: {df.shape}")
print(f"Classes: {list(data.target_names)}")

print("\nClass Distribution:")
print(df["target"].value_counts())

print(f"\nMissing Values: {df.isnull().sum().sum()}")

# ==========================================================
# STEP 2: Data Preprocessing
# ==========================================================

X = data.data
y = data.target

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# Feature Scaling (needed for Logistic Regression)
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ==========================================================
# STEP 3: Train Logistic Regression
# ==========================================================

print("\n--- Training Logistic Regression ---")

lr = LogisticRegression(
    max_iter=10000,
    random_state=42,
)

lr.fit(X_train_sc, y_train)

y_pred_lr = lr.predict(X_test_sc)

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"Accuracy : {acc_lr:.4f}")
print(f"Precision: {prec_lr:.4f}")
print(f"Recall   : {rec_lr:.4f}")
print(f"F1 Score : {f1_lr:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lr,
        target_names=data.target_names,
    )
)

# ==========================================================
# STEP 4: Train Random Forest
# ==========================================================

print("\n--- Training Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Accuracy : {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall   : {rec_rf:.4f}")
print(f"F1 Score : {f1_rf:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=data.target_names,
    )
)

# ==========================================================
# STEP 5: Compare Models
# ==========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Logistic Regression Accuracy : {acc_lr:.4f}")
print(f"Random Forest Accuracy       : {acc_rf:.4f}")

if acc_lr >= acc_rf:
    best_model = lr
    best_scaler = scaler
    best_name = "Logistic Regression"
    needs_scaling = True
else:
    best_model = rf
    best_scaler = None
    best_name = "Random Forest"
    needs_scaling = False

print(f"\nSelected Model: {best_name}")

# ==========================================================
# STEP 6: Save Model and Scaler
# ==========================================================

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(
    f"\nmodel.pkl saved - Size: "
    f"{os.path.getsize('model.pkl') / 1024:.1f} KB"
)

print(
    f"scaler.pkl saved - Size: "
    f"{os.path.getsize('scaler.pkl') / 1024:.1f} KB"
)

print("\nTraining complete!")
print("Run test_model.py to verify the saved model.")