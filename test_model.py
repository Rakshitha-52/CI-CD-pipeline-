import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Step 1: Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model loaded successfully:", type(model).__name__)

# Step 2: Prepare sample input data
data = load_breast_cancer()

X = data.data
y = data.target

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

sample_input = X_test[:3]

# Scale input (required for Logistic Regression)
sample_input_scaled = scaler.transform(sample_input)

# Step 3: Run predictions
predictions = model.predict(sample_input_scaled)
probabilities = model.predict_proba(sample_input_scaled)

# Step 4: Print results
class_names = data.target_names

print("\nPrediction Results")
print("-" * 60)

for i in range(3):
    print(
        f"Sample {i+1}: "
        f"True = {class_names[y_test[i]]} | "
        f"Predicted = {class_names[predictions[i]]} | "
        f"P(malignant) = {probabilities[i][0]:.4f} | "
        f"P(benign) = {probabilities[i][1]:.4f}"
    )

print(type(model).__name__)