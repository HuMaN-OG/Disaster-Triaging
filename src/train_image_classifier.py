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

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_datasets
from src.model import DisasterClassifier, build_model
from src.utils.image_classifier import (
    BACKBONE_LR,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    CLASSIFIER_LR,
    DATASET_ROOT,
    EARLY_STOPPING_PATIENCE,
    FREEZE_BACKBONE,
    GRAD_CLIP_MAX_NORM,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MIXUP_ALPHA,
    NUM_EPOCHS,
    NUM_WORKERS,
    PARTIAL_FINETUNE,
    RANDOM_SEED,
    USE_CLASS_WEIGHTS,
    USE_MIXUP,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    compute_class_weights,
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
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--classifier-lr", type=float, default=CLASSIFIER_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--finetune-mode",
        choices=("frozen", "partial", "full"),
        default=None,
        help="Training mode: frozen head-only, partial backbone, or full fine-tune.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Deprecated alias for --finetune-mode full.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted CrossEntropyLoss.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        help="Label smoothing epsilon for CrossEntropyLoss (0 = off).",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=MIXUP_ALPHA,
        help="MixUp Beta distribution alpha (0 = disabled).",
    )
    parser.add_argument(
        "--no-mixup",
        action="store_true",
        help="Disable MixUp augmentation.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=GRAD_CLIP_MAX_NORM,
        help="Gradient clipping max L2 norm (0 = disabled).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
        help="Number of linear LR warm-up epochs before cosine annealing.",
    )
    return parser.parse_args()


def resolve_finetune_mode(args: argparse.Namespace) -> str:
    if args.finetune_mode is not None:
        return args.finetune_mode
    if args.unfreeze_backbone:
        return "full"
    if PARTIAL_FINETUNE and not FREEZE_BACKBONE:
        return "partial"
    if PARTIAL_FINETUNE:
        return "partial"
    if FREEZE_BACKBONE:
        return "frozen"
    return "full"


def build_optimizer(
    model: DisasterClassifier,
    finetune_mode: str,
    args: argparse.Namespace,
) -> AdamW:
    """Build AdamW with differential LRs when backbone params are trainable."""
    backbone_params = model.backbone_trainable_parameters()
    classifier_params = model.classifier_trainable_parameters()

    if finetune_mode == "frozen":
        return AdamW(
            classifier_params,
            lr=args.classifier_lr,
            weight_decay=args.weight_decay,
        )

    param_groups: list[dict[str, Any]] = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay}
        )
    if classifier_params:
        param_groups.append(
            {
                "params": classifier_params,
                "lr": args.classifier_lr,
                "weight_decay": args.weight_decay,
            }
        )
    return AdamW(param_groups)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.4,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float, list[int], list[int]]:
    """
    Run one train or eval epoch.

    If ``optimizer`` is provided, gradients are computed and applied (train mode);
    otherwise the model is evaluated without updates.
    When ``use_mixup=True`` and in train mode, MixUp augmentation is applied.
    Gradient clipping is applied when ``grad_clip_norm > 0`` and in train mode.
    """
    is_train = optimizer is not None
    apply_mixup = is_train and use_mixup and mixup_alpha > 0
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

            if apply_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                index = torch.randperm(images.size(0), device=device)
                images = lam * images + (1.0 - lam) * images[index]
                labels_b = labels[index]
                logits = model(images)
                loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, train_samples, val_samples = build_datasets(root=args.dataset_root)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

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

    finetune_mode = resolve_finetune_mode(args)
    model = build_model(finetune_mode=finetune_mode, device=device)
    counts = model.trainable_parameter_counts()
    print(
        f"Finetune mode: {finetune_mode} | "
        f"trainable params: {counts['trainable']:,} / {counts['total']:,}"
    )

    use_class_weights = USE_CLASS_WEIGHTS and not args.no_class_weights
    label_smoothing = args.label_smoothing
    if use_class_weights:
        class_weights = compute_class_weights(train_samples).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"Using class-weighted CrossEntropyLoss (label_smoothing={label_smoothing}): {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = build_optimizer(model, finetune_mode, args)
    if finetune_mode == "frozen":
        print(f"Optimizer LR (classifier): {args.classifier_lr:.2e}")
    else:
        print(
            f"Optimizer LRs - backbone: {args.backbone_lr:.2e}, "
            f"classifier: {args.classifier_lr:.2e}"
        )
    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )
    use_mixup = USE_MIXUP and not args.no_mixup
    if use_mixup:
        print(f"MixUp augmentation enabled (alpha={args.mixup_alpha})")
    if args.grad_clip > 0:
        print(f"Gradient clipping enabled (max_norm={args.grad_clip})")
    print(
        f"LR schedule: {args.warmup_epochs}-epoch linear warmup + "
        f"CosineAnnealingLR (T_max={max(1, args.epochs - args.warmup_epochs)})"
    )

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
            model, train_loader, criterion, device, optimizer=optimizer,
            use_mixup=use_mixup, mixup_alpha=args.mixup_alpha,
            grad_clip_norm=args.grad_clip,
        )
        val_loss, val_acc, val_true, val_pred = run_epoch(
            model, val_loader, criterion, device, optimizer=None
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
                checkpoint_path,
                extra={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": CLASS_NAMES,
                    "history": history,
                    "finetune_mode": finetune_mode,
                },
            )
            print(f"  Saved best checkpoint -> {checkpoint_path}")
        else:
            epochs_without_improve += 1
            print(f"  No improvement ({epochs_without_improve}/{args.patience})")
            if epochs_without_improve >= args.patience:
                print("Early stopping triggered.")
                break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best epoch: {best_epoch}")

    model, checkpoint = DisasterClassifier.load(
        checkpoint_path, device=device, finetune_mode=finetune_mode
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
    print(f"Confusion matrix -> {cm_path}")
    print(f"Training history -> {hist_path}")

    return {
        "history": history,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path,
        "training_time_sec": elapsed,
        "trainable_params": counts["trainable"],
        "finetune_mode": finetune_mode,
    }


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
