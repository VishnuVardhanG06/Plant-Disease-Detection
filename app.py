"""
app.py
------
Flask web server for the Plant Disease Detection application.

Routes:
    GET  /         → Serves the main web UI (templates/index.html)
    POST /predict  → Accepts a leaf image, returns JSON prediction

Run with:
    py app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import json
import logging
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# ── Ensure project root is importable when running as `py app.py` ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_SAVE_PATH
from src.feature_extraction import extract_hog_features
from src.predict import load_model
from src.preprocess import preprocess_single

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ─── Load model once at startup ────────────────────────────────────────────────
logger.info("Loading model from: %s", MODEL_SAVE_PATH)
try:
    MODEL, CLASS_NAMES = load_model(MODEL_SAVE_PATH)
    logger.info("Model ready. Classes: %s", CLASS_NAMES)
except FileNotFoundError as e:
    logger.critical("%s\nRun 'py main.py' first to train the model.", e)
    sys.exit(1)

# ─── Load disease tips ─────────────────────────────────────────────────────────
TIPS_PATH = os.path.join(os.path.dirname(__file__), "disease_tips.json")
with open(TIPS_PATH, encoding="utf-8") as fh:
    DISEASE_TIPS: dict = json.load(fh)
logger.info("Disease tips loaded — %d entries.", len(DISEASE_TIPS))


# ─── Helper ────────────────────────────────────────────────────────────────────

def _format_display_name(raw_label: str) -> str:
    """Convert raw class name like 'Tomato___Early_blight' to 'Tomato Early Blight'."""
    return raw_label.replace("___", " - ").replace("_", " ").title()


def _run_inference(img_bgr: np.ndarray) -> dict:
    """
    Run the ML inference pipeline on a decoded BGR image.

    Args:
        img_bgr (np.ndarray): Image read by OpenCV (BGR channel order).

    Returns:
        dict with keys: disease, display_name, confidence, severity, tips
    """
    # Preprocess → HOG → predict
    img_preprocessed = preprocess_single(img_bgr)
    features = extract_hog_features(img_preprocessed).reshape(1, -1)

    raw_pred = MODEL.predict(features)[0]

    # ── DEBUG: log exactly what the model returns ──────────────────────────────
    logger.info("DEBUG raw_pred=%r  type=%s  CLASS_NAMES=%s",
                raw_pred, type(raw_pred).__name__, CLASS_NAMES)

    # Map integer index → class name string
    if CLASS_NAMES and hasattr(raw_pred, "__index__"):
        disease_label = str(CLASS_NAMES[int(raw_pred)])  # cast np.str_ → plain str
    else:
        disease_label = str(raw_pred)

    logger.info("DEBUG disease_label=%r", disease_label)


    # Confidence score
    confidence = 0.0
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(features)[0]
        confidence = round(float(proba.max() * 100), 1)

    # Fetch tips — fall back to _default if class not in tips file
    class_tips = DISEASE_TIPS.get(disease_label)           # None if not found
    fallback    = DISEASE_TIPS.get("_default", {})
    tips        = class_tips if class_tips else fallback

    # Always show a clean display name from the actual class label,
    # never the generic "Unknown Disease" string from _default
    display_name = (class_tips or {}).get("display_name") or _format_display_name(disease_label)
    severity     = tips.get("severity", "unknown")

    return {
        "disease":      disease_label,
        "display_name": display_name,
        "confidence":   confidence,
        "severity":     severity,
        "tips": {
            "description": tips.get("description", ""),
            "symptoms":    tips.get("symptoms", []),
            "prevention":  tips.get("prevention", []),
            "treatment":   tips.get("treatment", []),
        },
    }


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/debug")
def debug():
    """Return model class names and disease_tips keys for diagnostics."""
    return jsonify({
        "class_names_in_model": CLASS_NAMES,
        "class_names_count": len(CLASS_NAMES),
        "disease_tips_keys": list(DISEASE_TIPS.keys()),
    })


@app.route("/")

def index():
    """Serve the main single-page UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a leaf image via multipart/form-data and return a JSON prediction.

    Request:
        POST /predict
        Form field: 'image' (image file)

    Response (200):
        {
          "disease":      "Tomato___Early_blight",
          "display_name": "Tomato Early Blight",
          "confidence":   94.3,
          "severity":     "moderate",
          "tips": { "description": "...", "symptoms": [], "prevention": [], "treatment": [] }
        }

    Error responses (400 / 500):
        { "error": "Human-readable message" }
    """
    # ── Validate upload ────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file received. Please upload a leaf image."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # ── Read image bytes → OpenCV array ───────────────────────────────────────
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image. Please upload a valid JPG, PNG, or BMP file."}), 400

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        result = _run_inference(img_bgr)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500

    logger.info(
        "Prediction: %s  confidence=%.1f%%  severity=%s",
        result["disease"], result["confidence"], result["severity"]
    )
    return jsonify(result)


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
