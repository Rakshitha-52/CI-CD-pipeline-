# tests/test_model.py
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def test_model_predicts_known_samples_correctly():
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    data = load_breast_cancer()
    X, y = data.data, data.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_indices = [0, 5, 10, 20, 50]
    samples = X_test[sample_indices]
    true_labels = y_test[sample_indices]

    samples_scaled = scaler.transform(samples)
    predictions = model.predict(samples_scaled)

    n_correct = sum(true_labels == predictions)
    assert n_correct == len(sample_indices), f"Only {n_correct}/{len(sample_indices)} predictions correct"