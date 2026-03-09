"""
main.py
-------
Entry point for the Plant Disease Detection ML pipeline.

Run this script from the project root to execute the full workflow:
  1. Load dataset images and labels
  2. Preprocess images (resize → grayscale → normalize)
  3. Extract HOG feature vectors
  4. Encode string labels to integers
  5. Split dataset into train / test sets (stratified 80/20)
  6. Train Random Forest and SVM classifiers
  7. Evaluate both models and generate reports
  8. Save the best-performing model to trained_models/best_model.pkl

Usage:
    python main.py
"""

import logging
import sys
import time

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    MODEL_SAVE_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.data_loader import load_dataset
from src.evaluate_model import compare_and_save_best
from src.feature_extraction import extract_features_from_dataset
from src.preprocess import preprocess_images
from src.train_model import encode_labels, train_random_forest, train_svm

# ─── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Execute the full machine learning pipeline end-to-end.

    Raises:
        SystemExit: If no images are loaded from the dataset.
    """
    pipeline_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("  Plant Disease Detection — ML Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load Dataset ───────────────────────────────────────────────────
    logger.info("[Step 1/7] Loading dataset from: %s", DATASET_PATH)
    images, string_labels, class_names = load_dataset(DATASET_PATH)

    if not images:
        logger.critical(
            "No images loaded from '%s'. "
            "Ensure class sub-folders contain image files.", DATASET_PATH
        )
        sys.exit(1)

    # ── Step 2: Preprocess Images ──────────────────────────────────────────────
    logger.info("[Step 2/7] Preprocessing images (resize → grayscale → normalize)...")
    X_images = preprocess_images(images)
    del images  # Free raw image memory as soon as preprocessing is done

    # ── Step 3: Extract HOG Features ──────────────────────────────────────────
    logger.info("[Step 3/7] Extracting HOG features...")
    X = extract_features_from_dataset(X_images)
    del X_images  # Free preprocessed image memory

    # ── Step 4: Encode Labels ──────────────────────────────────────────────────
    logger.info("[Step 4/7] Encoding class labels...")
    y, label_encoder = encode_labels(string_labels)

    # ── Step 5: Train / Test Split ─────────────────────────────────────────────
    logger.info(
        "[Step 5/7] Splitting dataset  (train=%.0f%%  test=%.0f%%)...",
        (1 - TEST_SIZE) * 100, TEST_SIZE * 100,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,     # Preserve class distribution in both splits
    )
    logger.info("  Training samples : %d", len(X_train))
    logger.info("  Testing  samples : %d", len(X_test))

    # ── Step 6: Train Models ───────────────────────────────────────────────────
    logger.info("[Step 6/7] Training classifiers...")
    rf_model  = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    # ── Step 7: Evaluate and Save Best Model ───────────────────────────────────
    logger.info("[Step 7/7] Evaluating models and saving the best one...")
    models = {
        "Random Forest":         rf_model,
        "Support Vector Machine": svm_model,
    }
    ordered_class_names = list(label_encoder.classes_)

    best_name, _ = compare_and_save_best(
        models,
        X_test,
        y_test,
        class_names=ordered_class_names,
        save_path=MODEL_SAVE_PATH,
        reports_dir=REPORTS_DIR,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f s", elapsed)
    logger.info("  Best model  : %s", best_name)
    logger.info("  Model saved : %s", MODEL_SAVE_PATH)
    logger.info("  Reports     : %s/", REPORTS_DIR)
    logger.info("=" * 60)
    logger.info("To predict a new image run:")
    logger.info("  python src/predict.py path/to/leaf_image.jpg")


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
