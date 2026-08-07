"""
Shared configuration, metrics helpers, and plotting utilities for the
disaster image classifier.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# src/utils/image_classifier.py → project root is three levels up
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT: Path = PROJECT_ROOT / "dataset"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHECKPOINT_PATH: Path = MODELS_DIR / "disaster_classifier.pth"

# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Flood",
    "Earthquake",
    "Urban Fire",
    "Wild Fire",
    "Landslide",
    "Drought",
]
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Folder-path → training label mapping.
# Both Damaged_Infrastructure/Earthquake and Damaged_Infrastructure/Infrastructure
# map to "Earthquake". Matching uses exact path components (not substrings).
FOLDER_TO_CLASS: dict[str, str] = {
    "Water_Disaster": "Flood",
    "Earthquake": "Earthquake",
    "Infrastructure": "Earthquake",
    "Urban_Fire": "Urban Fire",
    "Wild_Fire": "Wild Fire",
    "Land_Slide": "Landslide",
    "Drought": "Drought",
}

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
)

# ---------------------------------------------------------------------------
# ImageNet normalisation (EfficientNet / torchvision defaults)
# ---------------------------------------------------------------------------
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_SIZE: int = 224

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.20
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 40
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
EARLY_STOPPING_PATIENCE: int = 8
NUM_WORKERS: int = 0  # Windows-friendly default; raise on Linux/macOS
RANDOM_SEED: int = 42
FREEZE_BACKBONE: bool = True
PARTIAL_FINETUNE: bool = True
UNFREEZE_LAST_N_BLOCKS: int = 5
BACKBONE_LR: float = 1e-5
CLASSIFIER_LR: float = 1e-4
USE_CLASS_WEIGHTS: bool = True

# ---------------------------------------------------------------------------
# Phase 2 — optimization hyper-parameters
# ---------------------------------------------------------------------------
LABEL_SMOOTHING: float = 0.05        # CrossEntropyLoss label smoothing epsilon
USE_MIXUP: bool = False              # MixUp data augmentation during training (postponed)
MIXUP_ALPHA: float = 0.4            # MixUp Beta distribution alpha
GRAD_CLIP_MAX_NORM: float = 1.0     # Gradient clipping max L2 norm (0 = off)
WARMUP_EPOCHS: int = 3              # Linear LR warm-up epochs before cosine


def compute_class_weights(
    samples: Sequence[tuple],
    num_classes: int = NUM_CLASSES,
) -> "torch.Tensor":
    """
    Compute balanced class weights for CrossEntropyLoss from training samples.

    Uses inverse-frequency weighting: weight_i = total / (num_classes * count_i).
    """
    import torch
    from collections import Counter

    counts = Counter(label for _, label in samples)
    total = sum(counts.values())
    weights = [
        total / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create models/ and results/ directories if they do not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, float | str | np.ndarray]:
    """
    Compute accuracy, precision, recall, F1, confusion matrix, and a text report.

    Precision / recall / F1 use macro averaging across classes.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred_arr, labels=list(range(len(class_names)))
        ),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=list(class_names),
            digits=4,
            zero_division=0,
        ),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    save_path: Path | None = None,
    title: str = "Confusion Matrix",
) -> Path:
    """Render and save a labelled confusion-matrix heatmap."""
    ensure_dirs()
    if save_path is None:
        save_path = RESULTS_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> Path:
    """
    Plot train/val loss and accuracy curves.

    Expected keys: train_loss, val_loss, train_acc, val_acc.
    """
    ensure_dirs()
    if save_path is None:
        save_path = RESULTS_DIR / "training_history.png"

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training History")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
