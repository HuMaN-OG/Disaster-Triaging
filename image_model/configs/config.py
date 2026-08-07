"""
Configuration dataclasses for the Image Authenticity Verification system.

Allows dynamic dataset mapping, flexible thresholds, and dataset-agnostic setup.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

_HERE = Path(__file__).resolve()
PROJECT_ROOT: Path = _HERE.parent.parent.parent  # image_model/configs/ → project root
IMAGE_MODEL_DIR: Path = _HERE.parent.parent      # image_model/
MODELS_DIR: Path = IMAGE_MODEL_DIR / "models"
RESULTS_DIR: Path = IMAGE_MODEL_DIR / "results"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"

FinetuneMode = Literal["frozen", "partial", "full"]
ModelName = Literal[
    "convnext_tiny", "swin_t", "efficientnet_v2_s",
    "efficientnet_b2", "efficientnet_b0", "vit_b_16"
]

# Default binary classes (can be overridden dynamically)
DEFAULT_CLASS_NAMES: list[str] = ["Real Photograph", "AI Generated"]

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class AuthenticityVerificationConfig:
    """
    Configurable parameters for the Image Authenticity Verification status module.
    """

    # Status thresholds (configurable, default values)
    flagged_threshold: float = 0.70    # AI prob >= 0.70 -> FLAGGED
    pass_threshold: float = 0.30       # AI prob <= 0.30 -> PASS (Real prob >= 0.70)
    # Between 0.30 and 0.70 -> UNCERTAIN


@dataclass
class AuthenticityConfig:
    """
    Central dataset-agnostic configuration for AI Image Detection model.
    """

    # ---- Dataset Mapping (Configurable) ----
    dataset_root: str = ""
    class_names: list[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    # Custom folder-to-class map (e.g., {"real_photos": "Real Photograph", "midjourney": "AI Generated"})
    folder_to_class_map: dict[str, str] = field(default_factory=dict)
    
    # Optional generator metadata mapping (e.g. {"midjourney_v6": "Midjourney"})
    generator_metadata_map: dict[str, str] = field(default_factory=dict)

    image_size: int = 224
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 0.15 inferred
    use_weighted_sampler: bool = True

    # ---- Architecture ----
    model_name: ModelName = "convnext_tiny"
    finetune_mode: FinetuneMode = "partial"
    unfreeze_last_n: int = 4
    pretrained: bool = True

    # ---- Training ----
    num_epochs: int = 40
    batch_size: int = 32
    num_workers: int = 0
    random_seed: int = 42

    # ---- Optimizer & LR ----
    backbone_lr: float = 5e-6
    classifier_lr: float = 5e-5
    weight_decay: float = 1e-4

    # ---- Loss & Augmentation ----
    use_class_weights: bool = True
    label_smoothing: float = 0.05
    # Mild augmentations to avoid destroying micro AI artifacts
    use_subtle_aug: bool = True

    # ---- LR Schedule ----
    warmup_epochs: int = 3
    eta_min: float = 1e-7

    # ---- Regularization ----
    grad_clip_max_norm: float = 1.0
    dropout: float = 0.3
    early_stopping_patience: int = 8

    # ---- Precision & Thresholds ----
    use_amp: bool = True
    verification_config: AuthenticityVerificationConfig = field(
        default_factory=AuthenticityVerificationConfig
    )

    # ---- Paths ----
    checkpoint_dir: str = str(CHECKPOINTS_DIR)
    best_model_path: str = str(MODELS_DIR / "authenticity_best_model.pth")
    results_dir: str = str(RESULTS_DIR)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Persist config to JSON for reproducibility."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "AuthenticityConfig":
        """Load config from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "verification_config" in data and isinstance(data["verification_config"], dict):
            data["verification_config"] = AuthenticityVerificationConfig(**data["verification_config"])
        return cls(**data)
