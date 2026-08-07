"""
Standalone evaluation script for image_model checkpoints.

Outputs classification reports, per-class accuracies, and confusion matrices
without touching legacy outputs in src/ or results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from image_model.configs import CLASS_NAMES, TrainingConfig
from image_model.datasets import DisasterImageDataset, build_dataloaders, get_val_transforms
from image_model.models import DisasterClassifier
from image_model.utils import compute_classification_metrics, plot_confusion_matrix, print_metrics_summary


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config: TrainingConfig | None = None,
    split: str = "test",
) -> dict:
    """Evaluate a saved 4-class checkpoint on train, val, or test splits."""
    if config is None:
        config = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = DisasterClassifier.load(checkpoint_path, device=device)
    model.eval()

    train_loader, val_loader, test_loader, split_info = build_dataloaders(config)

    if split == "train":
        loader = train_loader
    elif split == "val":
        loader = val_loader
    else:
        loader = test_loader

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, class_names=CLASS_NAMES)
    print_metrics_summary(metrics, split_name=f"4-Class Evaluation ({split.capitalize()} Split)")

    results_dir = Path(config.results_dir)
    cm_path = plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=results_dir / f"confusion_matrix_eval_{split}.png",
        title=f"Confusion Matrix ({split.capitalize()} Split)",
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "per_class_accuracy": metrics["per_class_accuracy"],
        },
        "confusion_matrix_path": str(cm_path),
    }

    report_path = results_dir / f"evaluation_report_{split}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 4-class image_model")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, split=args.split)
