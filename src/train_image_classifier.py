"""
Train the EfficientNet-B0 disaster image classifier.

Usage
-----
From the project root:

    python -m src.train_image_classifier
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_datasets
from src.model import DisasterClassifier, build_model
from src.utils.image_classifier import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DATASET_ROOT,
    EARLY_STOPPING_PATIENCE,
    FREEZE_BACKBONE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    WEIGHT_DECAY,
    compute_classification_metrics,
    ensure_dirs,
    plot_confusion_matrix,
    plot_training_history,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train disaster image classifier")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Train the full network instead of freezing the EfficientNet backbone.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to save the best model checkpoint.",
    )
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    """
    Run one train or eval epoch.

    If ``optimizer`` is provided, gradients are computed and applied (train mode);
    otherwise the model is evaluated without updates.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if is_train else "val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_true, all_pred


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, train_samples, val_samples = build_datasets(root=args.dataset_root)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    # Class distribution summary
    from collections import Counter

    train_counts = Counter(label for _, label in train_samples)
    val_counts = Counter(label for _, label in val_samples)
    for idx, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:12s}  train={train_counts.get(idx, 0):4d}  val={val_counts.get(idx, 0):4d}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    freeze_backbone = FREEZE_BACKBONE and not args.unfreeze_backbone
    model = build_model(freeze_backbone=freeze_backbone, device=device)
    counts = model.trainable_parameter_counts()
    print(
        f"Backbone frozen: {model.backbone_frozen} | "
        f"trainable params: {counts['trainable']:,} / {counts['total']:,}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    checkpoint_path = args.checkpoint

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc, val_true, val_pred = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            model.save(
                checkpoint_path,
                extra={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": CLASS_NAMES,
                    "history": history,
                },
            )
            print(f"  Saved best checkpoint → {checkpoint_path}")
        else:
            epochs_without_improve += 1
            print(f"  No improvement ({epochs_without_improve}/{args.patience})")
            if epochs_without_improve >= args.patience:
                print("Early stopping triggered.")
                break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best epoch: {best_epoch}")

    # Reload best weights for final validation metrics / plots
    model, checkpoint = DisasterClassifier.load(
        checkpoint_path, device=device, freeze_backbone=freeze_backbone
    )
    _, _, val_true, val_pred = run_epoch(model, val_loader, criterion, device, optimizer=None)
    metrics = compute_classification_metrics(val_true, val_pred)

    print("\n=== Best model — validation metrics ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(metrics["classification_report"])

    cm_path = plot_confusion_matrix(metrics["confusion_matrix"])
    hist_path = plot_training_history(history)
    print(f"Confusion matrix → {cm_path}")
    print(f"Training history → {hist_path}")

    return {
        "history": history,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path,
    }


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
