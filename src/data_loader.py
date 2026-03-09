"""
data_loader.py
--------------
Loads images and string labels from the PlantVillage dataset.

Each sub-folder inside the dataset root is treated as one disease class.
The folder name becomes the class label for every image inside it.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.config import DATASET_PATH, SUPPORTED_EXTENSIONS

# Module-level logger — inherits root logger config set in main.py
logger = logging.getLogger(__name__)


def _get_class_dirs(dataset_path: Path) -> List[Path]:
    """
    Return a sorted list of class sub-directory paths.

    Args:
        dataset_path (Path): Root dataset directory.

    Returns:
        List[Path]: Sorted list of class directory paths.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If no sub-directories are found.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}\n"
            "Make sure the 'dataset/' folder exists and contains class sub-folders."
        )

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class sub-folders found in: {dataset_path}")

    return class_dirs


def load_dataset(
    dataset_path: str = DATASET_PATH,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Walk the dataset directory, load images, and extract labels from folder names.

    Args:
        dataset_path (str): Path to the root dataset folder. Defaults to config value.

    Returns:
        images      (List[np.ndarray]): Raw BGR images loaded via OpenCV.
        labels      (List[str]):        Corresponding class label for each image.
        class_names (List[str]):        Sorted list of unique class labels.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If no sub-directories (classes) are found.
    """
    root = Path(dataset_path)
    class_dirs = _get_class_dirs(root)
    class_names: List[str] = [d.name for d in class_dirs]

    logger.info("Found %d classes: %s", len(class_names), class_names)

    images: List[np.ndarray] = []
    labels: List[str] = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        loaded_count = 0
        skipped_count = 0

        for img_path in class_dir.iterdir():
            # Skip files that are not recognised image formats
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise IOError("cv2.imread returned None")

                images.append(img)
                labels.append(class_name)
                loaded_count += 1

            except Exception as exc:
                logger.warning("Skipping corrupt/unreadable image '%s': %s", img_path.name, exc)
                skipped_count += 1

        logger.info(
            "  %-40s  loaded=%d  skipped=%d",
            f"'{class_name}'", loaded_count, skipped_count
        )

    logger.info("Total images loaded: %d", len(images))
    return images, labels, class_names
