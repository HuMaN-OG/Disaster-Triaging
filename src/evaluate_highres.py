"""
High-Resolution (384x384) Test-Time Augmentation (TTA) & Multi-Scale Evaluation.

Evaluates the top-performing benchmark architectures at higher spatial resolutions
(384x384) with multi-scale TTA to maximize visual detail extraction and boost accuracy
towards 88% - 90%.
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
from src.dataset import DisasterImageDataset, build_datasets
from src.utils.image_classifier import (
    CLASS_NAMES,
    DATASET_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    RESULTS_DIR,
    compute_classification_metrics,
    ensure_dirs,
    plot_confusion_matrix,
    set_seed,
)

HIGH_RES_SIZE: int = 384


def get_highres_tta_transforms(img_size: int = HIGH_RES_SIZE) -> list[transforms.Compose]:
    """
    Return high-resolution TTA evaluation transforms for target image size:
    1. Standard (Resize + CenterCrop + Normalize)
    2. Horizontally Flipped
    """
    standard = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    flipped = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return [standard, flipped]


def predict_highres_probs(
    model: BenchmarkClassifier,
    val_samples: list[tuple[Path, int]],
    device: torch.device,
    img_size: int = HIGH_RES_SIZE,
) -> np.ndarray:
    """Compute softmax predictions at target resolution with TTA."""
    tta_transforms = get_highres_tta_transforms(img_size)
    all_probs_list: list[np.ndarray] = []

    for tfm in tta_transforms:
        dataset = DisasterImageDataset(val_samples, transform=tfm)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        probs_batches: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"Predicting {img_size}x{img_size} ({model.model_name})", leave=False):
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_batches.append(probs)

        tfm_probs = np.vstack(probs_batches)
        all_probs_list.append(tfm_probs)

    return np.mean(all_probs_list, axis=0)


def main() -> None:
    set_seed(42)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, _, val_samples = build_datasets(root=DATASET_ROOT)
    y_true = [label for _, label in val_samples]
    print(f"Validation samples: {len(val_samples)}")

    target_models: list[tuple[ModelName, int]] = [
        ("swin_t", 384),
        ("convnext_tiny", 384),
        ("vit_b_16", 224),
        ("efficientnet_v2_s", 384),
    ]
    model_probs: dict[str, np.ndarray] = {}
    single_metrics: dict[str, dict[str, Any]] = {}

    for model_name, target_size in target_models:
        ckpt_path = MODELS_DIR / f"disaster_classifier_{model_name}.pth"
        if not ckpt_path.exists():
            continue

        print(f"[High-Res Evaluation] Loading {model_name} from {ckpt_path}")
        model, _ = BenchmarkClassifier.load(ckpt_path, device=device, finetune_mode="frozen")
        model.eval()

        probs = predict_highres_probs(model, val_samples, device, img_size=target_size)
        model_probs[model_name] = probs

        preds = probs.argmax(axis=1)
        m = compute_classification_metrics(y_true, preds)
        single_metrics[model_name] = m
        print(f"  {model_name:<18s} ({target_size}x{target_size} TTA) -> Accuracy: {m['accuracy']*100:.2f}%, F1: {m['f1']*100:.2f}%")

    if not model_probs:
        print("No models found!")
        return

    # Ensemble weighting: 0.35 Swin-T, 0.25 ConvNeXt, 0.20 ViT, 0.20 EffV2
    weights = {"swin_t": 0.35, "convnext_tiny": 0.25, "vit_b_16": 0.20, "efficientnet_v2_s": 0.20}
    total_w = sum(weights[m] for m in model_probs)

    ensemble_probs = sum((weights[m] / total_w) * model_probs[m] for m in model_probs)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    metrics = compute_classification_metrics(y_true, ensemble_preds)

    print("\n" + "=" * 60)
    print("HIGH-RESOLUTION (384x384) ENSEMBLE METRICS")
    print("=" * 60)
    print(f"High-Res Validation Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"High-Res Macro Precision    : {metrics['precision']*100:.2f}%")
    print(f"High-Res Macro Recall       : {metrics['recall']*100:.2f}%")
    print(f"High-Res Macro F1 Score     : {metrics['f1']*100:.2f}%")

    print("\nPer-Class Report:\n", metrics["classification_report"])

    cm_path = RESULTS_DIR / "confusion_matrix_highres_ensemble.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=cm_path,
        title="Confusion Matrix — High-Res 384x384 Multi-Model Ensemble",
    )
    print(f"High-Res Confusion Matrix saved -> {cm_path}")

    out_json = RESULTS_DIR / "highres_ensemble_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "highres_accuracy": metrics["accuracy"],
                "highres_precision": metrics["precision"],
                "highres_recall": metrics["recall"],
                "highres_f1": metrics["f1"],
                "classification_report": metrics["classification_report"],
            },
            f,
            indent=2,
        )
    print(f"High-Res results saved -> {out_json}")


if __name__ == "__main__":
    main()
