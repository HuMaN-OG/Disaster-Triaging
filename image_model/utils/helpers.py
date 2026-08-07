"""
Shared utilities for the image model pipeline.

Includes:
  - Reproducible seeding
  - Directory creation helpers
  - Class weight computation
  - Confusion matrix and training curve plotting
  - Pretty console reporting
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from image_model.configs import CLASS_NAMES, CHECKPOINTS_DIR, MODELS_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for full reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def ensure_dirs(
    extra_dirs: list[Path] | None = None,
) -> None:
    """Create models/, models/checkpoints/, and results/ if they don't exist."""
    for d in [MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if extra_dirs:
        for d in extra_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    """
    Compute full classification metrics.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1,
        per_class_accuracy, confusion_matrix, classification_report
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_classes = len(class_names)

    # Per-class accuracy
    per_class_acc: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        mask = y_true_arr == idx
        if mask.sum() > 0:
            per_class_acc[name] = float((y_pred_arr[mask] == idx).sum() / mask.sum())
        else:
            per_class_acc[name] = 0.0

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred_arr, labels=list(range(n_classes))
        ),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=list(class_names),
            digits=4,
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    save_path: Path | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Path:
    """
    Render and save a labelled confusion matrix heatmap.

    Parameters
    ----------
    cm:
        Confusion matrix array from sklearn.
    normalize:
        If True, normalize rows to show recall per class.
    """
    ensure_dirs()
    if save_path is None:
        save_path = RESULTS_DIR / "confusion_matrix.png"

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / np.maximum(row_sums, 1)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True", fontsize=11)
    axes[0].set_title(f"{title} (counts)")

    # Normalized (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.maximum(row_sums, 1)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True", fontsize=11)
    axes[1].set_title(f"{title} (normalized recall)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path | None = None,
    title: str = "Training History",
) -> Path:
    """
    Plot train/val loss and accuracy curves.

    Expected history keys: train_loss, val_loss, train_acc, val_acc.
    """
    ensure_dirs()
    if save_path is None:
        save_path = RESULTS_DIR / "training_history.png"

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3, linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="s", markersize=3, linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        label="Train",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    axes[1].plot(
        epochs,
        [a * 100 for a in history["val_acc"]],
        label="Val",
        marker="s",
        markersize=3,
        linewidth=1.5,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_class_distribution(
    split_info: dict[str, list],
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    """Print image count per class for each split."""
    print("\n📊 Class Distribution")
    header = f"  {'Class':<14}" + "".join(f"  {s.capitalize():<8}" for s in split_info)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, name in enumerate(class_names):
        row = f"  {name:<14}"
        for split_samples in split_info.values():
            count = sum(1 for _, lbl in split_samples if lbl == idx)
            row += f"  {count:<8}"
        print(row)
    totals = "  " + f"{'TOTAL':<14}" + "".join(
        f"  {len(s):<8}" for s in split_info.values()
    )
    print(totals)


def print_metrics_summary(metrics: dict, split_name: str = "Validation") -> None:
    """Pretty-print the key metrics from compute_classification_metrics."""
    print(f"\n📈 {split_name} Metrics")
    print(f"  Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 (macro): {metrics['f1'] * 100:.2f}%")
    if "per_class_accuracy" in metrics:
        print("\n  Per-class Accuracy:")
        for name, acc in metrics["per_class_accuracy"].items():
            bar = "█" * int(acc * 20)
            print(f"    {name:<14} {acc * 100:5.1f}%  {bar}")
    print()
