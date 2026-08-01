import os
import logging
import numpy as np
import joblib
from flask import Flask, request, jsonify
# nn App initialization nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
# __name__ tells Flask where to find templates and static files
app = Flask(__name__)
# nn Logging setup nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
# Logs every request and error to terminal — essential for debugging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
# nn Load model at startup (NOT inside routes) nnnnnnnnnnnnnnnnnnnnnnnnnnnnn
# Loading once at startup is efficient — avoids reloading on every request
MODEL_PATH = os.path.join('model', 'model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f'Model loaded: {type(model).__name__}')
    logger.info(f'Scaler loaded: {type(scaler).__name__}')
except FileNotFoundError as e:
    logger.error(f'Model file not found: {e}')
    logger.error('Run train_model.py first to generate model.pkl')
    raise SystemExit(1) # Stop the app — no point running without a model