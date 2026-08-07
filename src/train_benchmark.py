"""
Benchmark disaster image classifiers across multiple architectures.

Trains each model with identical dataset, split, augmentations, optimizer,
and evaluation metrics for fair comparison.

Usage
-----
    python -m src.train_benchmark
    python -m src.train_benchmark --models convnext_tiny vit_b_16 swin_t
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.benchmark_models import (
    BENCHMARK_MODELS,
    MODEL_DISPLAY_NAMES,
    BenchmarkClassifier,
    ModelName,
    build_benchmark_model,
)
from src.dataset import build_datasets
from src.train_image_classifier import build_optimizer, run_epoch
from src.utils.image_classifier import (
    BACKBONE_LR,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASSIFIER_LR,
    DATASET_ROOT,
    EARLY_STOPPING_PATIENCE,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    RESULTS_DIR,
    USE_CLASS_WEIGHTS,
    WEIGHT_DECAY,
    compute_class_weights,
    compute_classification_metrics,
    ensure_dirs,
    plot_confusion_matrix,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark disaster image classifiers")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BENCHMARK_MODELS,
        default=BENCHMARK_MODELS,
        help="Model architectures to benchmark.",
    )
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--classifier-lr", type=float, default=CLASSIFIER_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted CrossEntropyLoss.",
    )
    return parser.parse_args()


def train_single_model(
    model_name: ModelName,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_samples: list,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train one architecture and return validation metrics."""
    display = MODEL_DISPLAY_NAMES[model_name]
    print(f"\n{'=' * 60}")
    print(f"Training {display}")
    print(f"{'=' * 60}")

    model = build_benchmark_model(model_name=model_name, finetune_mode="partial", device=device)
    counts = model.trainable_parameter_counts()
    print(f"Trainable params: {counts['trainable']:,} / {counts['total']:,}")

    use_class_weights = USE_CLASS_WEIGHTS and not args.no_class_weights
    if use_class_weights:
        class_weights = compute_class_weights(train_samples).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    class _Args:
        pass

    opt_args = _Args()
    opt_args.classifier_lr = args.classifier_lr
    opt_args.backbone_lr = args.backbone_lr
    opt_args.weight_decay = args.weight_decay
    optimizer = build_optimizer(model, "partial", opt_args)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    checkpoint_path = MODELS_DIR / f"disaster_classifier_{model_name}.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_loss)
        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
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
                },
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print("Early stopping triggered.")
                break

    elapsed = time.time() - start

    model, _ = BenchmarkClassifier.load(
        checkpoint_path, device=device, finetune_mode="partial"
    )
    _, _, val_true, val_pred = run_epoch(model, val_loader, criterion, device, optimizer=None)
    metrics = compute_classification_metrics(val_true, val_pred)

    cm_path = plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=RESULTS_DIR / f"confusion_matrix_{model_name}.png",
        title=f"Confusion Matrix — {display}",
    )

    result = {
        "model": model_name,
        "display_name": display,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "trainable_params": counts["trainable"],
        "training_time_sec": elapsed,
        "training_time_min": elapsed / 60,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "confusion_matrix": str(cm_path),
    }

    print(f"\n{display} — val accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    return result


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print formatted comparison table."""
    header = (
        f"{'Model':<18} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
        f"{'Params':>12} {'Time(min)':>10}"
    )
    print("\n" + "=" * len(header))
    print("BENCHMARK COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['display_name']:<18} "
            f"{r['accuracy']*100:6.2f}% "
            f"{r['precision']*100:6.2f}% "
            f"{r['recall']*100:6.2f}% "
            f"{r['f1']*100:6.2f}% "
            f"{r['trainable_params']:>12,} "
            f"{r['training_time_min']:>10.1f}"
        )
    print("=" * len(header))


def recommend_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Recommend best model balancing accuracy and efficiency."""
    if not results:
        return {}

    best_acc = max(results, key=lambda x: x["accuracy"])
    # Efficiency score: accuracy / log10(trainable_params) — higher is better
    for r in results:
        import math

        r["efficiency_score"] = r["accuracy"] / math.log10(max(r["trainable_params"], 10))

    best_efficiency = max(results, key=lambda x: x["efficiency_score"])

    return {
        "best_accuracy": best_acc,
        "best_efficiency": best_efficiency,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, train_samples, val_samples = build_datasets(root=args.dataset_root)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

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

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        result = train_single_model(
            model_name, train_loader, val_loader, train_samples, device, args
        )
        results.append(result)

    print_comparison_table(results)

    rec = recommend_best(results)
    if rec:
        print(
            f"\nBest validation accuracy: {rec['best_accuracy']['display_name']} "
            f"({rec['best_accuracy']['accuracy']*100:.2f}%)"
        )
        print(
            f"Best accuracy/efficiency trade-off: {rec['best_efficiency']['display_name']} "
            f"(efficiency score: {rec['best_efficiency']['efficiency_score']:.4f})"
        )

    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "recommendation": rec}, f, indent=2)
    print(f"\nResults saved -> {results_path}")


if __name__ == "__main__":
    main()
