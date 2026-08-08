import os
import json
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from .config import AuthenticityConfig
from .logger import get_logger

logger = get_logger("authenticity.dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect_image_paths(config: AuthenticityConfig) -> Tuple[List[str], List[int]]:
    real_paths = []
    for sdir in config.real_subdirs:
        full_dir = os.path.join(config.dataset_dir, sdir)
        if os.path.exists(full_dir):
            for root, _, files in os.walk(full_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                        real_paths.append(os.path.join(root, f))
                        
    ai_paths = []
    full_ai_dir = os.path.join(config.dataset_dir, config.ai_subdir)
    if os.path.exists(full_ai_dir):
        for root, _, files in os.walk(full_ai_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                    ai_paths.append(os.path.join(root, f))
                    
    all_paths = real_paths + ai_paths
    all_labels = [0] * len(real_paths) + [1] * len(ai_paths)
    
    logger.info(f"Collected total images: Real={len(real_paths)}, AI={len(ai_paths)}, Total={len(all_paths)}")
    return all_paths, all_labels

def _fix_sample_path(path: str, config: AuthenticityConfig) -> str:
    if os.path.exists(path):
        return path
    # Try remapping drive / root path
    lower_path = path.lower()
    if "disaster-triaging" in lower_path:
        rel = path.lower().split("disaster-triaging")[-1].lstrip("\\/")
        remapped = os.path.join(config.root_dir, rel)
        if os.path.exists(remapped):
            return remapped
    return path

def get_or_create_splits(config: AuthenticityConfig, seed: int = 42) -> Dict[str, List[Tuple[str, int]]]:
    splits_file = os.path.join(config.data_dir, "splits.json")
    if os.path.exists(splits_file):
        logger.info(f"Loading existing dataset splits from {splits_file}")
        with open(splits_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            splits = {k: [(_fix_sample_path(item[0], config), item[1]) for item in v] for k, v in data.items()}
            return splits

    paths, labels = collect_image_paths(config)
    
    # 70% train, 15% val, 15% test stratified
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    
    splits = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }
    
    config.create_dirs()
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
        
    logger.info(f"Created & saved new dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits

class AuthenticityDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error reading image {path}: {e}, substituting black image.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, path

def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, eval_transform

def get_authenticity_dataloaders(config: AuthenticityConfig) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, float]]:
    splits = get_or_create_splits(config, seed=config.train.seed)
    train_transform, eval_transform = get_transforms(config.model.img_size)
    
    train_ds = AuthenticityDataset(splits["train"], transform=train_transform)
    val_ds = AuthenticityDataset(splits["val"], transform=eval_transform)
    test_ds = AuthenticityDataset(splits["test"], transform=eval_transform)
    
    # Calculate class counts for weighting
    train_labels = [s[1] for s in splits["train"]]
    real_count = sum(1 for l in train_labels if l == 0)
    ai_count = sum(1 for l in train_labels if l == 1)
    total = len(train_labels)
    
    # Class weights for Binary Cross Entropy (pos_weight = count(neg) / count(pos))
    pos_weight = float(real_count) / max(ai_count, 1)
    weights = {"real_count": real_count, "ai_count": ai_count, "pos_weight": pos_weight}
    
    logger.info(f"Train distribution: Real={real_count}, AI={ai_count}, Pos Weight={pos_weight:.3f}")
    
    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, weights
