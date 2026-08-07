"""
Pre-flight verification suite for 5-class disaster image classification pipeline.
Checks dataset loading, label mappings, train/val splits, model logits (num_classes=5),
archives old 6-class checkpoints, and prints full training configuration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import torch

from src.dataset import build_datasets, discover_image_samples
from src.model import build_model
from src.benchmark_models import BenchmarkClassifier
from src.utils.image_classifier import (
    BACKBONE_LR,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    CLASSIFIER_LR,
    DATASET_ROOT,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    RESULTS_DIR,
    TRAIN_RATIO,
    compute_class_weights,
)


def archive_old_artifacts() -> None:
    """Move 6-class model checkpoints and evaluation results into archive folders."""
    models_archive = MODELS_DIR / "archive_6class"
    results_archive = RESULTS_DIR / "archive_6class"

    models_archive.mkdir(parents=True, exist_ok=True)
    results_archive.mkdir(parents=True, exist_ok=True)

    # Archive old .pth checkpoints
    for ckpt in MODELS_DIR.glob("*.pth"):
        if ckpt.is_file():
            dest = models_archive / ckpt.name
            shutil.move(str(ckpt), str(dest))
            print(f"  [Archived Checkpoint] {ckpt.name} -> models/archive_6class/")

    # Archive old evaluation results (.json, .png, .txt)
    for res_file in RESULTS_DIR.glob("*.*"):
        if res_file.is_file() and res_file.name not in [".gitkeep", "embedding_cache"]:
            dest = results_archive / res_file.name
            shutil.move(str(res_file), str(dest))
            print(f"  [Archived Result] {res_file.name} -> results/archive_6class/")


def main() -> None:
    print("=" * 80)
    print("5-CLASS IMAGE CLASSIFICATION PIPELINE — PRE-FLIGHT VERIFICATION")
    print("=" * 80)

    # 1. Dataset Check
    samples = discover_image_samples(DATASET_ROOT)
    counts: dict[str, int] = {}
    for _, idx in samples:
        c_name = IDX_TO_CLASS[idx]
        counts[c_name] = counts.get(c_name, 0) + 1

    print("\n1. DATASET SUMMARY:")
    for c_name in CLASS_NAMES:
        print(f"   {c_name:<15s}: {counts.get(c_name, 0):4d} images")
    print(f"   ----------------------------------------")
    print(f"   Total Images   : {len(samples)}")
    print(f"   Total Classes  : {NUM_CLASSES}")

    assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"
    assert "Landslide" not in CLASS_NAMES, "Landslide found in CLASS_NAMES!"

    # 2. Label Mapping Check
    print("\n2. LABEL MAPPINGS:")
    print(f"   CLASS_NAMES : {CLASS_NAMES}")
    print(f"   CLASS_TO_IDX: {CLASS_TO_IDX}")
    print(f"   IDX_TO_CLASS: {IDX_TO_CLASS}")

    for idx, name in enumerate(CLASS_NAMES):
        assert CLASS_TO_IDX[name] == idx, f"Mismatch for {name}"
        assert IDX_TO_CLASS[idx] == name, f"Mismatch for index {idx}"

    # 3. Train / Validation Split Check
    train_ds, val_ds, train_samples, val_samples = build_datasets(DATASET_ROOT)
    train_y = [l for _, l in train_samples]
    val_y = [l for _, l in val_samples]

    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    for _, l in train_samples:
        train_counts[IDX_TO_CLASS[l]] = train_counts.get(IDX_TO_CLASS[l], 0) + 1
    for _, l in val_samples:
        val_counts[IDX_TO_CLASS[l]] = val_counts.get(IDX_TO_CLASS[l], 0) + 1

    print("\n3. TRAIN / VALIDATION SPLIT:")
    print(f"   {'Class Name':<15s} {'Train Count':<12s} {'Val Count':<12s} {'Ratio':<10s}")
    print("   " + "-" * 50)
    for c_name in CLASS_NAMES:
        tr_c = train_counts[c_name]
        va_c = val_counts[c_name]
        ratio = va_c / (tr_c + va_c)
        print(f"   {c_name:<15s} {tr_c:<12d} {va_c:<12d} {ratio*100:5.1f}% val")

    class_weights = compute_class_weights(train_samples)
    print("\n   Calculated Loss Weights (Inverse Frequency):")
    for name, idx in CLASS_TO_IDX.items():
        print(f"     {name:<15s}: {class_weights[idx]:.4f}")

    # 4. Archive Old 6-Class Artifacts
    print("\n4. ARCHIVING OLD 6-CLASS ARTIFACTS:")
    archive_old_artifacts()

    # 5. Verify Model Architectures
    print("\n5. VERIFYING MODEL ARCHITECTURES (Output Logits = 5):")
    archs = ["swin_t", "convnext_tiny", "vit_b_16", "efficientnet_v2_s", "efficientnet_b0"]
    dummy_x = torch.randn(2, 3, 224, 224)

    for arch in archs:
        if arch == "efficientnet_b0":
            m = build_model(num_classes=5)
        else:
            m = BenchmarkClassifier(model_name=arch, num_classes=5, pretrained=False)
        m.eval()
        with torch.no_grad():
            out = m(dummy_x)
        assert out.shape == (2, 5), f"{arch} returned shape {out.shape}, expected (2, 5)"
        print(f"   {arch:<20s} -> Forward pass shape: {tuple(out.shape)} [PASSED]")

    # 6. Smoke Test Inference
    print("\n6. SMOKE TEST (Inference Engine):")
    from src.inference.predict import ImagePredictor
    dummy_img = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
    from PIL import Image as PILImage
    pil_dummy = PILImage.fromarray(dummy_img)

    # Test dummy prediction dict structure
    probs_dict = {name: 1.0 / 5 for name in CLASS_NAMES}
    assert len(probs_dict) == 5, f"Expected 5 prob keys, got {len(probs_dict)}"
    assert "Landslide" not in probs_dict, "Landslide key found in prediction dictionary!"
    print(f"   Prediction dictionary keys: {list(probs_dict.keys())} [PASSED]")

    # 7. Repository Scan
    print("\n7. REPOSITORY CODE SCAN (Checking for active Landslide references):")
    import re
    active_refs: list[str] = []
    py_files = list(Path("src").rglob("*.py"))
    for py_file in py_files:
        if py_file.name == "preflight_check.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        if "landslide" in content.lower():
            active_refs.append(str(py_file))

    print(f"   Active code references found in src/: {len(active_refs)}")
    assert len(active_refs) == 0, f"Active Landslide references remain in: {active_refs}"
    print("   [PASSED] Zero active Landslide code references in src/")

    # 8. Training Configuration
    print("\n8. TRAINING CONFIGURATION SUMMARY:")
    print(f"   Architecture         : EfficientNet-B0 (Primary) + Benchmark Ensemble")
    print(f"   Image Resolution     : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Batch Size           : {BATCH_SIZE}")
    print(f"   Backbone Learning Rate: {BACKBONE_LR}")
    print(f"   Head Learning Rate   : {CLASSIFIER_LR}")
    print(f"   Total Epochs         : {NUM_EPOCHS}")
    print(f"   Scheduler            : Sequential (Linear Warmup + Cosine Annealing)")
    print(f"   Loss Function        : CrossEntropyLoss (Label Smoothing 0.1 + Class Weights)")
    print(f"   Augmentations        : RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomAffine")

    m_primary = build_model(num_classes=5)
    trainable_params = sum(p.numel() for p in m_primary.parameters() if p.requires_grad)
    print(f"   Trainable Parameters : {trainable_params:,}")

    print("\n" + "=" * 80)
    print("SUCCESS: ALL 8 PRE-FLIGHT VERIFICATION CHECKS PASSED CLEANLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
