"""
Evaluate a trained disaster image classifier checkpoint.

Usage
-----
From the project root:

    python -m src.evaluate
    python -m src.evaluate --checkpoint models/disaster_classifier.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DisasterImageDataset, build_datasets, get_val_transforms
from src.model import DisasterClassifier
from src.utils.image_classifier import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DATASET_ROOT,
    NUM_WORKERS,
    RESULTS_DIR,
    compute_classification_metrics,
    ensure_dirs,
    plot_confusion_matrix,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate disaster image classifier")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--split",
        choices=("val", "all"),
        default="val",
        help="Evaluate on the stratified validation split (default) or the full dataset.",
    )
    parser.add_argument(
        "--cm-path",
        type=str,
        default=str(RESULTS_DIR / "confusion_matrix.png"),
        help="Where to save the confusion-matrix figure.",
    )
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    """Return true labels, predictions, and average cross-entropy loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    return y_true, y_pred, avg_loss


def evaluate(args: argparse.Namespace) -> dict:
    set_seed()
    ensure_dirs()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model first "
            f"(python -m src.train_image_classifier)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model, checkpoint = DisasterClassifier.load(checkpoint_path, device=device)
    model.eval()

    train_ds, val_ds, train_samples, val_samples = build_datasets(root=args.dataset_root)

    if args.split == "val":
        dataset = val_ds
        print(f"Evaluating on validation split ({len(val_samples)} images)")
    else:
        all_samples = train_samples + val_samples
        dataset = DisasterImageDataset(all_samples, transform=get_val_transforms())
        print(f"Evaluating on full dataset ({len(all_samples)} images)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    y_true, y_pred, avg_loss = predict(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)

    print("\n=== Evaluation results ===")
    print(f"Loss     : {avg_loss:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(metrics["classification_report"])

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "val_acc" in checkpoint:
        print(f"Checkpoint recorded val_acc: {checkpoint['val_acc']:.4f}")

    cm_path = plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=Path(args.cm_path),
        title=f"Confusion Matrix ({args.split})",
    )
    print(f"Confusion matrix → {cm_path}")

    return {"loss": avg_loss, "metrics": metrics, "checkpoint": str(checkpoint_path)}


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
