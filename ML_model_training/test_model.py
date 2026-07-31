import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

print("Loading saved model...")

# Load the saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

print(f"Model type: {type(model).__name__}")  

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data (same split as training)
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Pick 5 mixed samples (some malignant, some benign)
sample_indices = [0, 5, 10, 20, 50]
samples = X_test[sample_indices]
true_labels = y_test[sample_indices]

# Apply the same scaling used during training
samples_scaled = scaler.transform(samples)

# Run predictions
predictions = model.predict(samples_scaled)
probabilities = model.predict_proba(samples_scaled)

# Display results
print("\n--- Prediction Results ---")
print(f'{"Sample":<8} {"True Label":<15} {"Predicted":<15} {"Confidence"}')
print("-" * 55)

for i in range(len(sample_indices)):
    true_name = data.target_names[true_labels[i]]
    pred_name = data.target_names[predictions[i]]
    confidence = max(probabilities[i]) * 100

    match = (
        "CORRECT"
        if true_labels[i] == predictions[i]
        else "WRONG"
    )

    print(
        f"{i + 1:<8} "
        f"{true_name:<15} "
        f"{pred_name:<15} "
        f"{confidence:.1f}% {match}"
    )

# Summary
n_correct = sum(true_labels == predictions)

print(f"\nVerification: {n_correct}/{len(sample_indices)} correct")

if n_correct == len(sample_indices):
    print("✅ All predictions correct — model.pkl is verified and ready!")
else:
    print("❌ Some predictions failed — check model training.")