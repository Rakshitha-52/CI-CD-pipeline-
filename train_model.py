import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_model(model_name, y_true, y_pred, target_names):
    """
    Display evaluation metrics for a model.
    """

    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names
    ))


def main():

    print("\nLoading Breast Cancer Dataset...\n")
    # 1. LOAD DATASET

    data = load_breast_cancer()

    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df["target"] = y

    print("Dataset Information")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Features: {len(data.feature_names)}")
    print(f"Classes: {list(data.target_names)}")

    print("\nClass Distribution:")
    print(df["target"].value_counts())

    
    # 2. DATA PREPROCESSING
    print("\nChecking Missing Values...")
    print(f"Missing Values: {df.isnull().sum().sum()}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.20,random_state=42,stratify=y
    )

    print("\nTrain-Test Split")
    print("-" * 30)
    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape : {X_test.shape}")

    
    # 3. FEATURE SCALING

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. LOGISTIC REGRESSION
    
    print("\nTraining Logistic Regression...")

    lr = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)

    evaluate_model(
        "Logistic Regression",
        y_test,
        y_pred_lr,
        data.target_names
    )

    lr_accuracy = accuracy_score(y_test, y_pred_lr)

    
    # 5. RANDOM FOREST

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    evaluate_model(
        "Random Forest",
        y_test,
        y_pred_rf,
        data.target_names
    )

    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    # 6. MODEL COMPARISON

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"Logistic Regression Accuracy : {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy       : {rf_accuracy:.4f}")

    if lr_accuracy >= rf_accuracy:
        best_model = lr
        best_model_name = "Logistic Regression"
    else:
        best_model = rf
        best_model_name = "Random Forest"

    print(f"\nBest Model Selected: {best_model_name}")

    
    # 7. SAVE MODEL

    print("\nSaving Model...")

    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    model_size = os.path.getsize("model.pkl") / 1024
    scaler_size = os.path.getsize("scaler.pkl") / 1024

    print("\nFiles Saved Successfully")
    print("-" * 30)
    print(f"model.pkl  : {model_size:.2f} KB")
    print(f"scaler.pkl : {scaler_size:.2f} KB")

    print("\nTraining Complete!")
    print("Artifacts generated:")
    print(" - model.pkl")
    print(" - scaler.pkl")


if __name__ == "__main__":
    main()