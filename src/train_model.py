"""
train_model.py
--------------
Trains two classifiers on the HOG feature matrix:
  1. Random Forest Classifier
  2. Support Vector Machine (SVC — RBF kernel)

Each training function logs wall-clock time to give a realistic
performance estimate on mid-range laptop hardware.

All hyperparameters are imported from src.config.
"""

import logging
import time
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.config import (
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
)

logger = logging.getLogger(__name__)


def encode_labels(labels: list) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string class labels into contiguous integer indices.

    Args:
        labels (list[str]): Raw string class labels from the dataset.

    Returns:
        encoded  (np.ndarray):   Integer-encoded label array.
        encoder  (LabelEncoder): Fitted encoder needed for inverse-transforming
                                 integer predictions back to class name strings.
    """
    encoder = LabelEncoder()
    encoded: np.ndarray = encoder.fit_transform(labels)
    logger.info("Label encoder classes: %s", list(encoder.classes_))
    return encoded, encoder


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = RF_N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the HOG feature matrix.

    Args:
        X_train      (np.ndarray): Training feature matrix, shape (N, F).
        y_train      (np.ndarray): Integer-encoded training labels, shape (N,).
        n_estimators (int):        Number of decision trees in the forest.
        random_state (int):        Seed for reproducibility.

    Returns:
        RandomForestClassifier: Fully trained model ready for evaluation.
    """
    logger.info("Training Random Forest  (n_estimators=%d) ...", n_estimators)
    t0 = time.perf_counter()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,          # Parallelise across all CPU cores
    )
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0
    logger.info("Random Forest training complete in %.1f s", elapsed)
    return model


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = SVM_KERNEL,
    C: float = SVM_C,
    gamma: str = SVM_GAMMA,
    random_state: int = RANDOM_STATE,
) -> SVC:
    """
    Train a Support Vector Machine classifier (SVC).

    Args:
        X_train      (np.ndarray): Training feature matrix, shape (N, F).
        y_train      (np.ndarray): Integer-encoded training labels, shape (N,).
        kernel       (str):        Kernel type — 'rbf' | 'linear' | 'poly'.
        C            (float):      Regularization parameter (higher = less regularization).
        gamma        (str|float):  Kernel coefficient — 'scale' adapts to feature variance.
        random_state (int):        Seed for reproducibility.

    Returns:
        SVC: Fully trained model with probability estimates enabled.
    """
    logger.info("Training SVM  (kernel=%s, C=%s, gamma=%s) ...", kernel, C, gamma)
    t0 = time.perf_counter()

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state,
        probability=True,   # Required for predict_proba() confidence scores
    )
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0
    logger.info("SVM training complete in %.1f s", elapsed)
    return model
