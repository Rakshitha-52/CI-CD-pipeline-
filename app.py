import os
import time
import logging
import numpy as np
import joblib
from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime
 
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
# Prediction Logging DB Setup
# ============================
# Moved out of predict() — this should run once at startup, not on
# every request. Note: on Cloud Run, this SQLite file is ephemeral
# and can be wiped when the container scales down/restarts. Fine for
# local testing and this stage of the project; a persistent store
# (Cloud SQL) is the real fix for production use.
 
DB_PATH = "predictions.db"
 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_time TEXT,
            features TEXT,
            prediction TEXT,
            confidence REAL,
            latency_ms REAL,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()
 
init_db()
 
def log_prediction(features, prediction, confidence, latency_ms):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (request_time, features, prediction, confidence, latency_ms, model_version) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            str(features),
            str(prediction),
            float(confidence),
            float(latency_ms),
            type(model).__name__,
        ),
    )
    conn.commit()
    conn.close()
 
# ============================
# Route 1: Root / Info Endpoint
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
# Route 1b: Lightweight Health Check
# ============================
# Separate from "/" — kept fast and dependency-free, no model logic,
# so Cloud Run's startup/liveness probes (and monitoring tools) have
# a minimal endpoint to hit.
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200
 
 
# ============================
# Route 2: Prediction
# ============================
 
@app.route("/predict", methods=["POST"])
def predict():
 
    start_time = time.time()
 
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
    confidence = float(probabilities[int(prediction)])
 
    latency_ms = (time.time() - start_time) * 1000
 
    # ----------------------------
    # Step 5: Log Prediction
    # ----------------------------
 
    try:
        log_prediction(features, predicted_class, confidence, latency_ms)
    except Exception as e:
        # Logging failures shouldn't break the actual prediction response
        logger.error(f"Failed to log prediction: {e}")
 
    # ----------------------------
    # Step 6: Build Response
    # ----------------------------
 
    response = {
        "prediction": predicted_class,
        "prediction_code": int(prediction),
        "probability_malignant": round(float(probabilities[0]), 4),
        "probability_benign": round(float(probabilities[1]), 4),
        "confidence": round(confidence, 4),
        "model": type(model).__name__,
        "latency_ms": round(latency_ms, 2),
        "status": "success"
    }
 
    logger.info(
        f"Prediction: {predicted_class} | Confidence: {response['confidence']} | Latency: {response['latency_ms']}ms"
    )
 
    return jsonify(response), 200
 
 
# ============================
# Route 3: Stats
# ============================
 
@app.route("/stats", methods=["GET"])
def stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT COUNT(*), AVG(confidence), AVG(latency_ms) FROM predictions"
    )
    count, avg_confidence, avg_latency = cursor.fetchone()
    conn.close()
 
    return jsonify({
        "total_predictions": count or 0,
        "average_confidence": round(avg_confidence, 4) if avg_confidence else None,
        "average_latency_ms": round(avg_latency, 2) if avg_latency else None
    }), 200
 
 
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
            "GET /health",
            "GET /stats",
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
 
