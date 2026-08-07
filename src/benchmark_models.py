"""
Multi-architecture disaster image classifiers for benchmark comparison.

All models share a common interface: partial fine-tuning, differential LR
param groups, and checkpoint save/load compatible with the training pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_V2_S_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
)

from src.utils.image_classifier import NUM_CLASSES, UNFREEZE_LAST_N_BLOCKS

FinetuneMode = Literal["frozen", "partial", "full"]
ModelName = Literal["efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s", "convnext_tiny", "vit_b_16", "swin_t"]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b_16": "ViT-B/16",
    "swin_t": "Swin-T",
}


class BenchmarkClassifier(nn.Module):
    """Wrapper that exposes a uniform fine-tuning API across architectures."""

    def __init__(
        self,
        model_name: ModelName,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        finetune_mode: FinetuneMode = "partial",
        unfreeze_last_n: int = UNFREEZE_LAST_N_BLOCKS,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self._unfreeze_last_n = unfreeze_last_n
        self._finetune_mode: FinetuneMode = "full"
        self._classifier_prefix = "head."

        if model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._classifier_prefix = "classifier."

        elif model_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._classifier_prefix = "classifier."

        elif model_name == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._classifier_prefix = "classifier."

        elif model_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier

        elif model_name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
            self._feature_module = self.backbone.encoder
            self._classifier_module = self.backbone.heads
            self._classifier_prefix = "heads."

        elif model_name == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_t(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.head
            self._classifier_prefix = "head."

        else:
            raise ValueError(f"Unknown model_name: {model_name!r}")

        self._apply_finetune_mode(finetune_mode)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self._classifier_module.parameters():
            param.requires_grad = True
        self._finetune_mode = "frozen"

    def partial_finetune(self, unfreeze_last_n: int | None = None) -> None:
        n = unfreeze_last_n if unfreeze_last_n is not None else self._unfreeze_last_n
        self._unfreeze_last_n = n

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self._classifier_module.parameters():
            param.requires_grad = True

        if self.model_name == "vit_b_16":
            layers = self.backbone.encoder.layers
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            feature_modules = list(self._feature_module.children())
            for module in feature_modules[-n:]:
                for param in module.parameters():
                    param.requires_grad = True

        self._finetune_mode = "partial"

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._finetune_mode = "full"

    def _apply_finetune_mode(self, mode: FinetuneMode) -> None:
        if mode == "frozen":
            self.freeze_backbone()
        elif mode == "partial":
            self.partial_finetune()
        elif mode == "full":
            self.unfreeze_backbone()
        else:
            raise ValueError(f"Unknown finetune_mode: {mode!r}")

    @property
    def finetune_mode(self) -> FinetuneMode:
        return self._finetune_mode

    @property
    def backbone_frozen(self) -> bool:
        return self._finetune_mode == "frozen"

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        classifier_ids = {id(p) for p in self._classifier_module.parameters()}
        return [
            p
            for p in self.backbone.parameters()
            if p.requires_grad and id(p) not in classifier_ids
        ]

    def classifier_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self._classifier_module.parameters() if p.requires_grad]

    def trainable_parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def save(self, path: str | Any, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "num_classes": NUM_CLASSES,
            "finetune_mode": self._finetune_mode,
            "backbone_frozen": self.backbone_frozen,
            "model_type": self.model_name,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Any,
        device: torch.device | str = "cpu",
        finetune_mode: FinetuneMode | None = None,
    ) -> tuple["BenchmarkClassifier", dict[str, Any]]:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)

        model_type = checkpoint.get("model_type", "efficientnet_b0")
        mode: FinetuneMode = checkpoint.get("finetune_mode", "partial")
        if finetune_mode is not None:
            mode = finetune_mode

        model = cls(
            model_name=model_type,
            num_classes=int(checkpoint.get("num_classes", NUM_CLASSES)),
            pretrained=False,
            finetune_mode=mode,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)
        return model, checkpoint


def build_benchmark_model(
    model_name: ModelName,
    finetune_mode: FinetuneMode = "partial",
    device: torch.device | str | None = None,
) -> BenchmarkClassifier:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BenchmarkClassifier(model_name=model_name, finetune_mode=finetune_mode)
    return model.to(device)


BENCHMARK_MODELS: list[ModelName] = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
]
