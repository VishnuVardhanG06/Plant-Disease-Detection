"""
preprocess.py
-------------
Image preprocessing pipeline:
  1. Resize images to a fixed square dimension (default 128×128)
  2. Convert BGR images to grayscale
  3. Normalize pixel values to the [0.0, 1.0] range

All hyperparameters are imported from src.config.
"""

import logging
from typing import List

import cv2
import numpy as np

from src.config import IMG_SIZE

logger = logging.getLogger(__name__)


def preprocess_single(image: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a single BGR image.

    Steps performed:
        1. Resize to IMG_SIZE using INTER_AREA interpolation (best for downscaling).
        2. Convert from BGR to grayscale.
        3. Normalize pixel values to float32 in [0.0, 1.0].

    Args:
        image (np.ndarray): Raw BGR image of any resolution.

    Returns:
        np.ndarray: Preprocessed float32 grayscale image of shape IMG_SIZE.
    """
    # Step 1 — Resize (width, height order for cv2.resize)
    resized = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)

    # Step 2 — Grayscale conversion (no-op if already single-channel)
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Step 3 — Normalize to [0.0, 1.0]
    normalized = gray.astype(np.float32) / 255.0
    return normalized


def preprocess_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Apply the preprocessing pipeline to a list of raw BGR images.

    Pre-allocates a single NumPy array up front to avoid repeated
    memory reallocation (more efficient than building a Python list).

    Args:
        images (List[np.ndarray]): List of raw BGR images read by OpenCV.

    Returns:
        np.ndarray: Preprocessed grayscale image array of shape (N, H, W)
                    with float32 values in [0.0, 1.0].
    """
    n = len(images)
    h, w = IMG_SIZE

    # Pre-allocate output array — avoids list-then-convert overhead
    output = np.empty((n, h, w), dtype=np.float32)

    for i, img in enumerate(images):
        output[i] = preprocess_single(img)

    logger.info("Preprocessed %d images → shape %s, dtype %s", n, output.shape, output.dtype)
    return output
