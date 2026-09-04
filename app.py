import os
import logging
import numpy as np
import joblib
from flask import Flask, request, jsonify

# ============================
# App Initialization
# ============================

app = Flask(__name__)

# ============================
# Logging Setup
# ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# ============================
# Load Model and Scaler
# ============================

MODEL_PATH = os.path.join("model", "model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Scaler loaded: {type(scaler).__name__}")

except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    logger.error("Run train_model.py first to generate model.pkl")
    raise SystemExit(1)

# ============================
# Route 1: Health Check
# ============================

@app.route("/", methods=["GET"])
def health_check():
    logger.info("Health check called")

    return jsonify({
        "status": "running",
        "model": type(model).__name__,
        "accuracy": "98.25%",
        "endpoint": "POST /predict with JSON body {'features': [30 floats]}",
        "dataset": "Wisconsin Breast Cancer Dataset"
    }), 200


# ============================
# Route 2: Prediction
# ============================

@app.route("/predict", methods=["POST"])
def predict():

    logger.info("Prediction request received")

    # ----------------------------
    # Step 1: Parse JSON Request
    # ----------------------------

    data = request.json

    # ----------------------------
    # Step 2: Validate Input
    # ----------------------------

    if data is None:
        logger.warning("No JSON body received")
        return jsonify({
            "error": "Request body must be JSON",
            "hint": "Set Content-Type: application/json"
        }), 400

    if "features" not in data:
        logger.warning("Missing features key in request")
        return jsonify({
            "error": "Missing 'features' key in JSON body",
            "hint": "Body must be: {'features': [f1, f2, ..., f30]}"
        }), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({
            "error": "'features' must be a list of numbers"
        }), 400

    if len(features) != 30:
        return jsonify({
            "error": f"Expected 30 features, got {len(features)}",
            "hint": "Breast Cancer dataset requires exactly 30 numerical features"
        }), 400

    # ----------------------------
    # Step 3: Preprocess Features
    # ----------------------------

    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

    except (ValueError, TypeError) as e:
        logger.error(f"Feature preprocessing failed: {e}")

        return jsonify({
            "error": f"Invalid feature values: {str(e)}"
        }), 400

    # ----------------------------
    # Step 4: Prediction
    # ----------------------------

    prediction = model.predict(arr_scaled)[0]
    probabilities = model.predict_proba(arr_scaled)[0]

    class_names = ["malignant", "benign"]
    predicted_class = class_names[int(prediction)]

    # ----------------------------
    # Step 5: Build Response
    # ----------------------------

    response = {
        "prediction": predicted_class,
        "prediction_code": int(prediction),
        "probability_malignant": round(float(probabilities[0]), 4),
        "probability_benign": round(float(probabilities[1]), 4),
        "confidence": round(float(probabilities[int(prediction)]), 4),
        "model": type(model).__name__,
        "status": "success"
    }

    logger.info(
        f"Prediction: {predicted_class} | Confidence: {response['confidence']}"
    )

    return jsonify(response), 200


# ============================
# STAGE 4: Error Handlers
# ============================

# ----------------------------
# Global Error Handler
# ----------------------------
# Catches any unhandled exception and returns a JSON response
# instead of Flask's default HTML error page.

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")

    return jsonify({
        "error": "Internal server error",
        "message": str(e),
        "status": "error"
    }), 500


# ----------------------------
# 404 Error Handler
# ----------------------------
# Returns JSON for unknown routes instead of HTML.

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available": [
            "GET /",
            "POST /predict"
        ]
    }), 404


# ============================
# App Entry Point
# ============================
# Google Cloud Run sets the PORT environment variable.
# Default to port 8080 when running locally.



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    

    logger.info(f"Starting Flask app on port {port}")

    app.run(
        host="0.0.0.0",   # Listen on all network interfaces
        port=port,
        debug=False       # Never enable debug mode in production
    )