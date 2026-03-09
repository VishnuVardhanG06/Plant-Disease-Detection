"""
evaluate_model.py
-----------------
Evaluates trained classifiers and produces:
  - Accuracy score
  - Full classification report  (precision, recall, F1-score per class)
  - Confusion matrix visualisation
  - Model accuracy comparison bar chart

Saves all charts to the reports/ directory (config.REPORTS_DIR).
Saves the best-performing model to trained_models/best_model.pkl via pickle.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import MODEL_SAVE_PATH, REPORTS_DIR

logger = logging.getLogger(__name__)


def _ensure_reports_dir(reports_dir: str = REPORTS_DIR) -> None:
    """Create the reports directory if it does not already exist."""
    os.makedirs(reports_dir, exist_ok=True)


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_names: List[str],
    reports_dir: str = REPORTS_DIR,
) -> float:
    """
    Evaluate one trained classifier and save its confusion matrix.

    Args:
        model       (Any):           Trained sklearn-compatible classifier.
        X_test      (np.ndarray):    Test feature matrix, shape (N, F).
        y_test      (np.ndarray):    True integer-encoded labels, shape (N,).
        model_name  (str):           Human-readable model name (used in chart titles).
        class_names (List[str]):     Ordered list of class label strings.
        reports_dir (str):           Directory where the PNG is saved.

    Returns:
        float: Test accuracy in the range [0.0, 1.0].
    """
    _ensure_reports_dir(reports_dir)

    y_pred: np.ndarray = model.predict(X_test)
    acc: float = accuracy_score(y_test, y_pred)

    # ── Console metrics ────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  Model    : %s", model_name)
    logger.info("  Accuracy : %.2f%%", acc * 100)
    logger.info("=" * 55)

    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info("Classification Report:\n%s", report)

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation="vertical")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    chart_path = os.path.join(reports_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)  # Free memory; don't block script with an interactive window

    logger.info("Confusion matrix saved → %s", chart_path)
    return acc


def _plot_accuracy_comparison(
    accuracies: Dict[str, float],
    reports_dir: str = REPORTS_DIR,
) -> None:
    """
    Save a bar chart comparing model accuracies.

    Args:
        accuracies  (Dict[str, float]): Mapping of {model_name: accuracy}.
        reports_dir (str):              Output directory.
    """
    names = list(accuracies.keys())
    values = [v * 100 for v in accuracies.values()]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    colors = palette[: len(names)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    ax.set_ylim(0, 115)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "model_accuracy_comparison.png")
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Accuracy comparison chart saved → %s", chart_path)


def compare_and_save_best(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    save_path: str = MODEL_SAVE_PATH,
    reports_dir: str = REPORTS_DIR,
) -> Tuple[str, Any]:
    """
    Evaluate all models, generate charts, and save the best model to disk.

    Args:
        models      (Dict[str, Any]):  Mapping of {model_name: trained_model}.
        X_test      (np.ndarray):      Test feature matrix.
        y_test      (np.ndarray):      True integer-encoded labels.
        class_names (List[str]):       Ordered class label strings.
        save_path   (str):             File path to save the winning model pickle.
        reports_dir (str):             Directory for chart outputs.

    Returns:
        best_model_name (str): Name of the best-performing model.
        best_model      (Any): The best-performing trained model object.
    """
    accuracies: Dict[str, float] = {}

    for model_name, model in models.items():
        acc = evaluate_single_model(
            model, X_test, y_test, model_name, class_names, reports_dir
        )
        accuracies[model_name] = acc

    # ── Accuracy comparison chart ──────────────────────────────────────────────
    _plot_accuracy_comparison(accuracies, reports_dir)

    # ── Select best model ──────────────────────────────────────────────────────
    best_name: str = max(accuracies, key=accuracies.get)
    best_model = models[best_name]
    logger.info(
        "Best model: '%s'  accuracy=%.2f%%", best_name, accuracies[best_name] * 100
    )

    # ── Persist best model ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Bundle the model with its class list so any consumer (Flask, CLI)
    # can map integer predictions back to human-readable class names.
    save_data = {
        "model": best_model,
        "classes": class_names,   # ordered list matching label-encoder indices
    }
    with open(save_path, "wb") as fh:
        pickle.dump(save_data, fh)
    logger.info("Best model saved → %s  (classes=%s)", save_path, class_names)

    return best_name, best_model
