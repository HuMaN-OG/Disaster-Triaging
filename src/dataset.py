"""
Custom PyTorch Dataset for disaster image classification.

Recursively scans ``dataset/``, maps folder paths to the six training labels
(including Infrastructure → Earthquake), and does **not** rely on ImageFolder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.image_classifier import (
    CLASS_TO_IDX,
    DATASET_ROOT,
    FOLDER_TO_CLASS,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
    TRAIN_RATIO,
)


def resolve_class_from_path(image_path: Path) -> str:
    """
    Infer the disaster class label from an image's folder path.

    Both ``Damaged_Infrastructure/Earthquake`` and
    ``Damaged_Infrastructure/Infrastructure`` map to ``Earthquake`` via
    ``FOLDER_TO_CLASS``. Matching uses exact path components so that
    ``Damaged_Infrastructure`` is not confused with ``Infrastructure``.
    """
    parts = set(image_path.parts)
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        if folder_name in parts:
            return class_name
    raise ValueError(f"Cannot resolve disaster class for path: {image_path}")


def discover_image_samples(root: Path) -> list[tuple[Path, int]]:
    """
    Recursively find image files under ``root`` and return (path, class_idx) pairs.

    Files whose parent folders cannot be mapped are skipped with a warning print.
    """
    samples: list[tuple[Path, int]] = []
    skipped = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            class_name = resolve_class_from_path(path)
            samples.append((path.resolve(), CLASS_TO_IDX[class_name]))
        except ValueError as exc:
            skipped += 1
            print(f"[dataset] Skipping {path}: {exc}")

    if not samples:
        raise FileNotFoundError(f"No labelled images found under {root}")

    if skipped:
        print(f"[dataset] Skipped {skipped} unmapped file(s).")

    return samples


def get_train_transforms() -> transforms.Compose:
    """Augmentation pipeline for training images."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Deterministic preprocessing for validation / evaluation images."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class DisasterImageDataset(Dataset):
    """
    Custom dataset that returns ``(image_tensor, class_idx)``.

    Parameters
    ----------
    samples:
        List of ``(image_path, class_idx)`` tuples.
    transform:
        Optional torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_train_val_split(
    samples: list[tuple[Path, int]],
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Stratified 80/20 (by default) train/validation split."""
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    return train_samples, val_samples


def build_datasets(
    root: Path | None = None,
) -> tuple[DisasterImageDataset, DisasterImageDataset, list[tuple[Path, int]], list[tuple[Path, int]]]:
    """
    Discover samples, stratify-split, and wrap in train/val Datasets.

    Returns
    -------
    train_ds, val_ds, train_samples, val_samples
    """
    dataset_root = Path(root) if root is not None else DATASET_ROOT
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    all_samples = discover_image_samples(dataset_root)
    train_samples, val_samples = stratified_train_val_split(all_samples)

    train_ds = DisasterImageDataset(train_samples, transform=get_train_transforms())
    val_ds = DisasterImageDataset(val_samples, transform=get_val_transforms())
    return train_ds, val_ds, train_samples, val_samples
