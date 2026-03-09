"""
feature_extraction.py
---------------------
Extracts Histogram of Oriented Gradients (HOG) features from preprocessed
grayscale images.

HOG captures the distribution of gradient orientations in local image regions,
making it a robust texture descriptor for visual classification tasks such as
plant disease detection.

All HOG hyperparameters are imported from src.config.
"""

import logging
from typing import Optional

import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from src.config import (
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
)

logger = logging.getLogger(__name__)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract a HOG feature vector from a single grayscale image.

    Args:
        image (np.ndarray): Preprocessed float32 grayscale image, shape (H, W).

    Returns:
        np.ndarray: 1-D HOG feature vector.
    """
    feature_vector: np.ndarray = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True,
    )
    return feature_vector


def extract_features_from_dataset(images: np.ndarray) -> np.ndarray:
    """
    Extract HOG feature vectors for every image in the dataset.

    Determines the feature length from the first image, then pre-allocates
    the full feature matrix — avoiding repeated list appends and the final
    np.array() conversion.

    Args:
        images (np.ndarray): Preprocessed grayscale images, shape (N, H, W).

    Returns:
        np.ndarray: Feature matrix of shape (N, feature_length), dtype float64.
    """
    n = len(images)

    # Compute the HOG vector for the first image to know the feature length
    first_vec = extract_hog_features(images[0])
    feature_len = first_vec.shape[0]

    logger.info(
        "HOG config — orientations=%d, pixels_per_cell=%s, cells_per_block=%s",
        HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    )
    logger.info("HOG feature vector length per image: %d", feature_len)

    # Pre-allocate the full output matrix
    feature_matrix = np.empty((n, feature_len), dtype=np.float64)
    feature_matrix[0] = first_vec  # Store the already-computed first vector

    # Extract features for remaining images with a tqdm progress bar
    for i, img in enumerate(tqdm(images[1:], desc="Extracting HOG features",
                                 unit="img", initial=1, total=n), start=1):
        feature_matrix[i] = extract_hog_features(img)

    logger.info("HOG feature matrix shape: %s", feature_matrix.shape)
    return feature_matrix
