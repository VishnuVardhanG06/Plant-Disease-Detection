"""
config.py
---------
Central configuration file for the Plant Disease Detection project.

All hyperparameters, file paths, and constants are defined here.
Import this module in other source files instead of hardcoding values.
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
# Root path of the project (two levels up from this file: src/ -> project root)
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH: str   = os.path.join(PROJECT_ROOT, "dataset")
MODEL_SAVE_PATH: str = os.path.join(PROJECT_ROOT, "trained_models", "best_model.pkl")
REPORTS_DIR: str    = os.path.join(PROJECT_ROOT, "reports")

# ─── Image Preprocessing ───────────────────────────────────────────────────────
IMG_SIZE: tuple = (128, 128)   # (height, width) — target resize dimensions

# ─── HOG Feature Extraction ────────────────────────────────────────────────────
HOG_ORIENTATIONS: int        = 9        # Gradient orientation bins
HOG_PIXELS_PER_CELL: tuple   = (8, 8)  # Cell size in pixels
HOG_CELLS_PER_BLOCK: tuple   = (2, 2)  # Block size in cells (for normalization)
HOG_BLOCK_NORM: str          = "L2-Hys"  # Block normalization method

# ─── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE: float    = 0.2   # Fraction of data used for testing (80/20 split)
RANDOM_STATE: int   = 42    # Global seed for reproducibility

# ─── Random Forest ─────────────────────────────────────────────────────────────
RF_N_ESTIMATORS: int = 200  # Number of trees

# ─── Support Vector Machine ────────────────────────────────────────────────────
SVM_KERNEL: str   = "linear"    # Kernel type
SVM_C: float      = 10.0     # Regularization parameter
SVM_GAMMA: str    = "scale"  # Kernel coefficient

# ─── Supported image formats ───────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".bmp")
