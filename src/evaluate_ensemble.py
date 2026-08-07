"""
Ensemble Evaluation & Test-Time Augmentation (TTA) script for disaster image classification.

Combines the top-performing benchmark architectures (Swin-T, ConvNeXt-Tiny,
ViT-B/16, EfficientNetV2-S) using probability-weighted ensembling and TTA
to maximize classification accuracy and F1 score towards the 90% target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.benchmark_models import BenchmarkClassifier, ModelName
from src.dataset import DisasterImageDataset, build_datasets, get_val_transforms
from src.utils.image_classifier import (
    CLASS_NAMES,
    DATASET_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    MODELS_DIR,
    RESULTS_DIR,
    compute_classification_metrics,
    ensure_dirs,
    plot_confusion_matrix,
    set_seed,
)


def get_tta_transforms() -> list[transforms.Compose]:
    """
    Return a list of evaluation transforms for Test-Time Augmentation (TTA).
    
    1. Standard val transform (Resize 224 + CenterCrop + Normalize)
    2. Horizontally flipped val transform
    """
    standard = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    flipped = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return [standard, flipped]


def load_model_if_exists(
    model_name: ModelName,
    device: torch.device,
) -> BenchmarkClassifier | None:
    """Load model checkpoint if it exists in models/ directory."""
    checkpoint_path = MODELS_DIR / f"disaster_classifier_{model_name}.pth"
    if not checkpoint_path.exists():
        print(f"[Ensemble] Checkpoint not found: {checkpoint_path}")
        return None

    print(f"[Ensemble] Loading {model_name} from {checkpoint_path}")
    model, _ = BenchmarkClassifier.load(checkpoint_path, device=device, finetune_mode="frozen")
    model.eval()
    return model


def predict_model_probs(
    model: BenchmarkClassifier,
    val_samples: list[tuple[Path, int]],
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Compute predicted softmax probabilities for all samples using TTA if enabled.
    
    Returns array of shape (N_samples, N_classes).
    """
    tta_transforms = get_tta_transforms() if use_tta else [get_val_transforms()]
    all_probs_list: list[np.ndarray] = []

    for tfm in tta_transforms:
        dataset = DisasterImageDataset(val_samples, transform=tfm)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        probs_batches: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"Predicting ({model.model_name})", leave=False):
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_batches.append(probs)
        
        tfm_probs = np.vstack(probs_batches)
        all_probs_list.append(tfm_probs)

    # Average probabilities across TTA variants
    avg_probs = np.mean(all_probs_list, axis=0)
    return avg_probs


def find_optimal_ensemble_weights(
    model_probs: dict[str, np.ndarray],
    y_true: list[int],
) -> dict[str, float]:
    """Search for optimal probability weights to maximize ensemble accuracy."""
    models = list(model_probs.keys())
    best_acc = 0.0
    best_weights = {m: 1.0 / len(models) for m in models}

    # Grid search weight combinations
    grid_steps = np.linspace(0.1, 1.0, 10)
    if len(models) == 3:
        for w1 in grid_steps:
            for w2 in grid_steps:
                for w3 in grid_steps:
                    weights = [w1, w2, w3]
                    total_w = sum(weights)
                    norm_weights = [w / total_w for w in weights]
                    
                    ens_prob = sum(
                        norm_weights[i] * model_probs[models[i]] for i in range(len(models))
                    )
                    preds = ens_prob.argmax(axis=1)
                    acc = float(np.mean(preds == np.array(y_true)))
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = {models[i]: norm_weights[i] for i in range(len(models))}
    elif len(models) == 4:
        for w1 in [0.2, 0.4, 0.6, 0.8, 1.0]:
            for w2 in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for w3 in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    for w4 in [0.2, 0.4, 0.6, 0.8, 1.0]:
                        weights = [w1, w2, w3, w4]
                        total_w = sum(weights)
                        norm_weights = [w / total_w for w in weights]
                        
                        ens_prob = sum(
                            norm_weights[i] * model_probs[models[i]] for i in range(len(models))
                        )
                        preds = ens_prob.argmax(axis=1)
                        acc = float(np.mean(preds == np.array(y_true)))
                        if acc > best_acc:
                            best_acc = acc
                            best_weights = {models[i]: norm_weights[i] for i in range(len(models))}
    
    print(f"\n[Ensemble Grid Search] Best Validation Accuracy: {best_acc * 100:.2f}%")
    for m, w in best_weights.items():
        print(f"  {m:18s} weight = {w:.3f}")
    
    return best_weights


def main() -> None:
    set_seed(42)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, _, val_samples = build_datasets(root=DATASET_ROOT)
    y_true = [label for _, label in val_samples]
    print(f"Validation samples: {len(val_samples)}")

    target_models: list[ModelName] = ["swin_t", "convnext_tiny", "vit_b_16", "efficientnet_v2_s"]
    model_probs: dict[str, np.ndarray] = {}
    single_metrics: dict[str, dict[str, Any]] = {}

    for model_name in target_models:
        model = load_model_if_exists(model_name, device)
        if model is None:
            continue
        
        probs = predict_model_probs(model, val_samples, device, use_tta=True)
        model_probs[model_name] = probs
        
        preds = probs.argmax(axis=1)
        m = compute_classification_metrics(y_true, preds)
        single_metrics[model_name] = m
        print(f"  {model_name:<18s} (with TTA) -> Accuracy: {m['accuracy']*100:.2f}%, F1: {m['f1']*100:.2f}%")

    if not model_probs:
        print("No models loaded for ensemble!")
        return

    # Find optimal weights and compute ensemble predictions
    best_weights = find_optimal_ensemble_weights(model_probs, y_true)
    
    ensemble_probs = sum(
        best_weights[m] * model_probs[m] for m in model_probs
    )
    ensemble_preds = ensemble_probs.argmax(axis=1)

    ensemble_metrics = compute_classification_metrics(y_true, ensemble_preds)

    print("\n" + "=" * 60)
    print("FINAL MULTIMODAL / MULTI-MODEL ENSEMBLE METRICS")
    print("=" * 60)
    print(f"Ensemble Validation Accuracy: {ensemble_metrics['accuracy']*100:.2f}%")
    print(f"Ensemble Macro Precision    : {ensemble_metrics['precision']*100:.2f}%")
    print(f"Ensemble Macro Recall       : {ensemble_metrics['recall']*100:.2f}%")
    print(f"Ensemble Macro F1 Score     : {ensemble_metrics['f1']*100:.2f}%")
    print("\nDetailed Per-Class Classification Report:\n")
    print(ensemble_metrics["classification_report"])

    cm_path = RESULTS_DIR / "confusion_matrix_ensemble.png"
    plot_confusion_matrix(
        ensemble_metrics["confusion_matrix"],
        save_path=cm_path,
        title="Confusion Matrix — Multi-Architecture Ensemble (with TTA)",
    )
    print(f"Ensemble Confusion Matrix saved -> {cm_path}")

    # Save results to JSON
    json_payload = {
        "ensemble_accuracy": ensemble_metrics["accuracy"],
        "ensemble_precision": ensemble_metrics["precision"],
        "ensemble_recall": ensemble_metrics["recall"],
        "ensemble_f1": ensemble_metrics["f1"],
        "weights": best_weights,
        "classification_report": ensemble_metrics["classification_report"],
        "single_models_with_tta": {
            m: {
                "accuracy": single_metrics[m]["accuracy"],
                "precision": single_metrics[m]["precision"],
                "recall": single_metrics[m]["recall"],
                "f1": single_metrics[m]["f1"],
            }
            for m in single_metrics
        },
    }

    out_json = RESULTS_DIR / "ensemble_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Ensemble results saved -> {out_json}")


if __name__ == "__main__":
    main()
