"""
Training loop implementation for 4-class disaster image classification.

Supports:
- Transfer learning with ImageNet weights
- AMP (Automatic Mixed Precision) via torch.cuda.amp
- CosineAnnealingLR with linear warmup
- Early stopping based on validation loss
- Gradient clipping
- Saving best model weights to image_model/models/best_model_4class.pth
- Saving checkpoints to image_model/models/checkpoints/

This file is part of the isolated image_model/ package and does not modify src/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_model.configs import CLASS_NAMES, TrainingConfig
from image_model.datasets import build_dataloaders, compute_class_weights
from image_model.models import DisasterClassifier, build_model
from image_model.utils import (
    compute_classification_metrics,
    ensure_dirs,
    plot_confusion_matrix,
    plot_training_history,
    print_class_distribution,
    print_metrics_summary,
    set_seed,
)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
) -> tuple[float, float, list[int], list[int]]:
    """Run a single training or validation epoch with AMP support."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    amp_enabled = use_amp and device.type == "cuda"

    with context:
        for images, labels in tqdm(loader, desc="train" if is_train else "val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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


def train(config: TrainingConfig | None = None) -> dict[str, Any]:
    """Train the model based on TrainingConfig."""
    if config is None:
        config = TrainingConfig()

    set_seed(config.random_seed)

    # Output paths dedicated to image_model package
    package_models_dir = Path(config.checkpoint_dir)
    package_results_dir = Path(config.results_dir)
    ensure_dirs([package_models_dir, package_results_dir])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | AMP Enabled: {config.use_amp and device.type == 'cuda'}")

    train_loader, val_loader, test_loader, split_info = build_dataloaders(config)
    print_class_distribution(split_info)

    model = build_model(
        model_name=config.model_name,
        num_classes=len(CLASS_NAMES),
        pretrained=config.pretrained,
        finetune_mode=config.finetune_mode,
        unfreeze_last_n=config.unfreeze_last_n,
        dropout=config.dropout,
        device=device,
    )
    counts = model.trainable_parameter_counts()
    print(
        f"Model: {config.model_name} | Mode: {config.finetune_mode} | "
        f"Trainable Params: {counts['trainable']:,} / {counts['total']:,}"
    )

    if config.use_class_weights:
        class_weights = compute_class_weights(split_info["train"], len(CLASS_NAMES)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
        print(f"Using class weights: {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # Optimizer
    param_groups = model.param_groups(
        backbone_lr=config.backbone_lr,
        classifier_lr=config.classifier_lr,
        weight_decay=config.weight_decay,
    )
    optimizer = AdamW(param_groups)

    # Scheduler
    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.num_epochs - config.warmup_epochs),
        eta_min=config.eta_min,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[config.warmup_epochs],
    )

    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    best_model_path = Path(config.best_model_path)

    start = time.time()
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=config.use_amp,
            grad_clip_norm=config.grad_clip_max_norm,
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, device, optimizer=None, use_amp=config.use_amp
        )

        scheduler.step()
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
                best_model_path,
                extra={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": CLASS_NAMES,
                    "history": history,
                    "config": config.to_dict(),
                },
            )
            print(f"  Saved best checkpoint -> {best_model_path}")
        else:
            epochs_without_improve += 1
            print(f"  No improvement ({epochs_without_improve}/{config.early_stopping_patience})")
            if epochs_without_improve >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best epoch: {best_epoch}")

    # Load best model for test evaluation
    best_model, checkpoint = DisasterClassifier.load(best_model_path, device=device)
    test_loss, test_acc, test_true, test_pred = run_epoch(
        best_model, test_loader, criterion, device, optimizer=None, use_amp=config.use_amp
    )

    metrics = compute_classification_metrics(test_true, test_pred)
    print_metrics_summary(metrics, split_name="Test Set (Unseen)")

    cm_path = plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=package_results_dir / "confusion_matrix_4class.png",
        title=f"Confusion Matrix (4-Class {config.model_name})",
    )
    hist_path = plot_training_history(
        history,
        save_path=package_results_dir / "training_history_4class.png",
        title=f"Training History (4-Class {config.model_name})",
    )

    summary = {
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "per_class_accuracy": metrics["per_class_accuracy"],
        },
        "training_time_sec": elapsed,
        "artifacts": {
            "best_model": str(best_model_path),
            "confusion_matrix": str(cm_path),
            "training_history": str(hist_path),
        },
    }

    summary_json_path = package_results_dir / "training_summary_4class.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train()
