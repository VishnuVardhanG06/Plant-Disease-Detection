"""
predict.py
----------
Loads a saved model from disk and predicts the disease class for a
given input leaf image.

Usage (from the project root directory):
    python src/predict.py path/to/leaf_image.jpg
    python src/predict.py path/to/leaf_image.jpg --model trained_models/best_model.pkl
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Ensure project root is on sys.path when script is run directly ─────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MODEL_SAVE_PATH, REPORTS_DIR
from src.feature_extraction import extract_hog_features
from src.preprocess import preprocess_single

logger = logging.getLogger(__name__)


# ─── Model I/O ─────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_SAVE_PATH):
    """
    Load a pickled model from disk.

    Supports two pkl formats:
      - **New** (dict): ``{"model": sklearn_model, "classes": ["cls_a", ...]}``
      - **Legacy** (plain sklearn model): returned as-is with empty class list.

    Args:
        model_path (str): Path to the .pkl file.

    Returns:
        Tuple[Any, List[str]]: ``(model, class_names)``

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run 'py main.py' first to train and save the model."
        )
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, dict):
        model = payload["model"]
        class_names = payload.get("classes", [])
    else:
        # Legacy format — plain sklearn model, no class name list
        model = payload
        class_names = []

    logger.info("Model loaded from: %s  |  classes: %s", model_path, class_names)
    return model, class_names


# ─── Inference Pipeline ────────────────────────────────────────────────────────

def predict_disease(image_path: str, model_path: str = MODEL_SAVE_PATH) -> str:
    """
    End-to-end prediction pipeline for a single leaf image.

    Steps:
        1. Load the trained model from disk.
        2. Read and validate the input image.
        3. Preprocess the image (resize → grayscale → normalize).
        4. Extract HOG features.
        5. Predict the disease class.
        6. Log confidence score if the model supports probability estimates.
        7. Save a labelled result image to reports/.

    Args:
        image_path (str): Path to the input leaf image file.
        model_path (str): Path to the saved model pickle.

    Returns:
        str: Predicted disease class label.
    """
    # Step 1 — Load model and class name list
    model, class_names = load_model(model_path)

    # Step 2 — Read image
    logger.info("Processing image: %s", image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image — check the file is a valid image: {image_path}")

    # Step 3 — Preprocess (resize → grayscale → normalize)
    img_preprocessed = preprocess_single(img_bgr)

    # Step 4 — HOG feature extraction
    features = extract_hog_features(img_preprocessed).reshape(1, -1)

    # Step 5 — Predict (returns integer index or string depending on encoder)
    raw_pred = model.predict(features)[0]

    # Map integer index → class name string if class list is available
    if class_names and hasattr(raw_pred, '__index__'):
        predicted_label: str = class_names[int(raw_pred)]
    else:
        predicted_label = str(raw_pred)

    # Step 6 — Confidence score
    confidence_str = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = proba.max() * 100
        confidence_str = f" (Confidence: {confidence:.1f}%)"

    logger.info("Predicted disease: %s%s", predicted_label, confidence_str)

    # Step 7 — Save labelled result image
    _save_prediction_image(img_bgr, predicted_label, confidence_str)

    return predicted_label


def _save_prediction_image(
    img_bgr: np.ndarray,
    label: str,
    confidence_str: str = "",
) -> None:
    """
    Save the original image with prediction label as a PNG to reports/.

    Args:
        img_bgr        (np.ndarray): Original BGR image read by OpenCV.
        label          (str):        Predicted class name.
        confidence_str (str):        Formatted confidence string, e.g. " (Confidence: 94.3%)".
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(
        f"Prediction: {label}{confidence_str}",
        fontsize=13, fontweight="bold", color="darkgreen",
    )
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "prediction_result.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Result image saved → %s", out_path)


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Plant Disease Detector — classify a leaf image using the trained model.",
    )
    # Positional argument: no flag needed
    parser.add_argument(
        "image",
        type=str,
        metavar="IMAGE_PATH",
        help="Path to the leaf image file (e.g., dataset/Tomato___healthy/img.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_SAVE_PATH,
        metavar="MODEL_PATH",
        help=f"Path to the trained model pickle (default: {MODEL_SAVE_PATH})",
    )
    return parser


if __name__ == "__main__":
    # Configure basic logging when running as a standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()
    predict_disease(args.image, args.model)
