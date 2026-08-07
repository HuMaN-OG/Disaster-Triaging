"""
Abstract Dataset Interface and Dataset Builder for Image Authenticity Verification.

Designed to be dataset-agnostic: does NOT assume folder structures, hardcoded names,
or dataset roots. Folders, labels, and metadata mappings are configured at runtime.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from image_model.configs import IMAGENET_MEAN, IMAGENET_STD, AuthenticityConfig

MIN_FILE_BYTES: int = 2_000
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


# ---------------------------------------------------------------------------
# Abstract Sample Specification
# ---------------------------------------------------------------------------

class AuthenticitySample:
    """Dataclass representing a single image sample with optional generator metadata."""

    def __init__(
        self,
        path: Path,
        label_idx: int,
        label_name: str,
        generator_source: str | None = None,
    ) -> None:
        self.path = path
        self.label_idx = label_idx
        self.label_name = label_name
        self.generator_source = generator_source or "unknown"

    def __repr__(self) -> str:
        return f"AuthenticitySample(path={self.path.name}, label={self.label_name}, generator={self.generator_source})"


# ---------------------------------------------------------------------------
# Dynamic Discovery Engine (Dataset-Agnostic)
# ---------------------------------------------------------------------------

def discover_authenticity_samples(
    root: Path,
    folder_to_class_map: dict[str, str],
    class_to_idx: dict[str, int],
    generator_metadata_map: dict[str, str] | None = None,
) -> list[AuthenticitySample]:
    """
    Recursively discover samples under root using user-provided folder mappings.

    Does not hardcode 'real' or 'ai_generated'. Matching relies entirely on
    `folder_to_class_map`. Optional `generator_metadata_map` preserves source info.
    """
    if generator_metadata_map is None:
        generator_metadata_map = {}

    samples: list[AuthenticitySample] = []
    skipped_unmapped = 0
    skipped_small = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.stat().st_size < MIN_FILE_BYTES:
            skipped_small += 1
            continue

        parts = set(path.parts)
        matched_class = None

        # Resolve class from custom map
        for folder_key, class_name in folder_to_class_map.items():
            if folder_key in parts:
                matched_class = class_name
                break

        if matched_class is None or matched_class not in class_to_idx:
            skipped_unmapped += 1
            continue

        # Resolve generator source if metadata exists
        gen_source = "unknown"
        for gen_key, gen_name in generator_metadata_map.items():
            if gen_key in parts:
                gen_source = gen_name
                break

        samples.append(
            AuthenticitySample(
                path=path.resolve(),
                label_idx=class_to_idx[matched_class],
                label_name=matched_class,
                generator_source=gen_source,
            )
        )

    if not samples:
        raise ValueError(
            f"No matching images found under {root} using map: {folder_to_class_map}"
        )

    if skipped_unmapped:
        print(f"[dataset] Ignored {skipped_unmapped} unmapped/unmatched file(s).")
    if skipped_small:
        print(f"[dataset] Ignored {skipped_small} file(s) below {MIN_FILE_BYTES} bytes.")

    return samples


# ---------------------------------------------------------------------------
# Data Transforms (Subtle Augmentation for AI Artifact Preservation)
# ---------------------------------------------------------------------------

def get_authenticity_train_transforms(image_size: int = 224, subtle: bool = True) -> transforms.Compose:
    """
    Augmentation pipeline tailored for AI vs Real classification.

    Avoids heavy blur or extreme distortion that destroys high-frequency AI artifacts.
    """
    if subtle:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_authenticity_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic validation / evaluation transforms."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class AuthenticityDataset(Dataset):
    """
    PyTorch Dataset returning (image_tensor, label_idx, generator_source).
    """

    def __init__(
        self,
        samples: list[AuthenticitySample],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            warnings.warn(f"[dataset] Error loading {sample.path}: {exc}. Using blank placeholder.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label_idx, sample.generator_source


# ---------------------------------------------------------------------------
# Dynamic DataLoader Builder
# ---------------------------------------------------------------------------

def build_authenticity_dataloaders(
    config: AuthenticityConfig,
    samples: list[AuthenticitySample] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[AuthenticitySample]]]:
    """
    Build 70/15/15 DataLoaders from samples or configured dataset root.
    """
    if samples is None:
        root = Path(config.dataset_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Dataset root '{root}' not found. Please provide a valid dataset_root in config."
            )
        class_to_idx = {name: idx for idx, name in enumerate(config.class_names)}
        samples = discover_authenticity_samples(
            root=root,
            folder_to_class_map=config.folder_to_class_map,
            class_to_idx=class_to_idx,
            generator_metadata_map=config.generator_metadata_map,
        )

    labels = [s.label_idx for s in samples]

    train_samples, temp_samples = train_test_split(
        samples, train_size=config.train_ratio, stratify=labels, random_state=config.random_seed
    )

    temp_labels = [s.label_idx for s in temp_samples]
    val_ratio_adj = config.val_ratio / (1.0 - config.train_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples, train_size=val_ratio_adj, stratify=temp_labels, random_state=config.random_seed
    )

    train_tf = get_authenticity_train_transforms(config.image_size, config.use_subtle_aug)
    val_tf = get_authenticity_val_transforms(config.image_size)

    train_ds = AuthenticityDataset(train_samples, transform=train_tf)
    val_ds = AuthenticityDataset(val_samples, transform=val_tf)
    test_ds = AuthenticityDataset(test_samples, transform=val_tf)

    if config.use_weighted_sampler:
        from collections import Counter
        counts = Counter(s.label_idx for s in train_samples)
        total = len(train_samples)
        weights = {lbl: total / cnt for lbl, cnt in counts.items()}
        sample_weights = [weights[s.label_idx] for s in train_samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    split_info = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    return train_loader, val_loader, test_loader, split_info
