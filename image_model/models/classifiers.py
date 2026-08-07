"""
Multi-architecture classifier factory for Image Authenticity Verification.

Supported backbones:
  convnext_tiny     — ConvNeXt Tiny (RECOMMENDED for spatial AI artifacts)
  swin_t            — Swin Transformer Tiny
  efficientnet_v2_s — EfficientNetV2-Small
  efficientnet_b2   — EfficientNet-B2
  efficientnet_b0   — EfficientNet-B0
  vit_b_16          — Vision Transformer B/16
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

from image_model.configs import DEFAULT_CLASS_NAMES

FinetuneMode = Literal["frozen", "partial", "full"]
ModelName = Literal[
    "convnext_tiny",
    "swin_t",
    "efficientnet_v2_s",
    "efficientnet_b2",
    "efficientnet_b0",
    "vit_b_16",
]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-T",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b_16": "ViT-B/16",
}

_DROPOUT: dict[str, float] = {
    "convnext_tiny": 0.3,
    "swin_t": 0.3,
    "efficientnet_v2_s": 0.3,
    "efficientnet_b2": 0.3,
    "efficientnet_b0": 0.2,
    "vit_b_16": 0.3,
}


class DisasterClassifier(nn.Module):
    """
    Unified multi-architecture classifier.

    Parameters
    ----------
    model_name:
        Architecture key (e.g. ``"convnext_tiny"``).
    num_classes:
        Number of output classes (default: 2 for binary Authenticity Verification).
    pretrained:
        Use ImageNet pretrained weights.
    finetune_mode:
        ``"frozen"`` — head only; ``"partial"`` — last N blocks + head;
        ``"full"`` — all parameters.
    unfreeze_last_n:
        Blocks to unfreeze in ``"partial"`` mode.
    dropout:
        Dropout rate inserted before the final linear layer.
    """

    def __init__(
        self,
        model_name: ModelName = "convnext_tiny",
        num_classes: int = 2,
        pretrained: bool = True,
        finetune_mode: FinetuneMode = "partial",
        unfreeze_last_n: int = 4,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self._num_classes = num_classes
        self._unfreeze_last_n = unfreeze_last_n
        self._finetune_mode: FinetuneMode = "full"
        drop = dropout if dropout is not None else _DROPOUT.get(model_name, 0.3)

        if model_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # LayerNorm
                self.backbone.classifier[1],  # Flatten
                nn.Dropout(p=drop),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._use_vit_layer_style = False

        elif model_name == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_t(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.head
            self._use_vit_layer_style = False

        elif model_name == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=drop, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._use_vit_layer_style = False

        elif model_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=drop, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._use_vit_layer_style = False

        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=drop, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.features
            self._classifier_module = self.backbone.classifier
            self._use_vit_layer_style = False

        elif model_name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(in_features, num_classes),
            )
            self._feature_module = self.backbone.encoder
            self._classifier_module = self.backbone.heads
            self._use_vit_layer_style = True

        else:
            raise ValueError(f"Unknown model_name: {model_name!r}")

        self._apply_finetune_mode(finetune_mode)

    def freeze_backbone(self) -> None:
        """Freeze backbone; only classifier head is trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self._classifier_module.parameters():
            param.requires_grad = True
        self._finetune_mode = "frozen"

    def partial_finetune(self, unfreeze_last_n: int | None = None) -> None:
        """Unfreeze last N feature blocks + classifier head."""
        n = unfreeze_last_n if unfreeze_last_n is not None else self._unfreeze_last_n

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self._classifier_module.parameters():
            param.requires_grad = True

        if self._use_vit_layer_style:
            layers = list(self.backbone.encoder.layers.children())
        else:
            layers = list(self._feature_module.children())

        for module in layers[-n:]:
            for param in module.parameters():
                param.requires_grad = True

        self._finetune_mode = "partial"

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
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

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        classifier_ids = {id(p) for p in self._classifier_module.parameters()}
        return [
            p for p in self.backbone.parameters()
            if p.requires_grad and id(p) not in classifier_ids
        ]

    def classifier_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self._classifier_module.parameters() if p.requires_grad]

    def param_groups(
        self,
        backbone_lr: float,
        classifier_lr: float,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        groups = []
        backbone_params = self.backbone_trainable_parameters()
        classifier_params = self.classifier_trainable_parameters()

        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        if classifier_params:
            groups.append({"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay})
        return groups

    def trainable_parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def save(self, path: str | Any, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "model_type": self.model_name,
            "num_classes": self._num_classes,
            "finetune_mode": self._finetune_mode,
            "class_names": DEFAULT_CLASS_NAMES,
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
    ) -> tuple["DisasterClassifier", dict[str, Any]]:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)

        model_name: ModelName = checkpoint.get("model_name") or checkpoint.get("model_type", "convnext_tiny")
        num_classes: int = int(checkpoint.get("num_classes", 2))
        mode: FinetuneMode = finetune_mode or checkpoint.get("finetune_mode", "partial")

        model = cls(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            finetune_mode=mode,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint


def build_model(
    model_name: ModelName = "convnext_tiny",
    num_classes: int = 2,
    pretrained: bool = True,
    finetune_mode: FinetuneMode = "partial",
    unfreeze_last_n: int = 4,
    dropout: float | None = None,
    device: torch.device | str | None = None,
) -> DisasterClassifier:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisasterClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        finetune_mode=finetune_mode,
        unfreeze_last_n=unfreeze_last_n,
        dropout=dropout,
    )
    return model.to(device)


BENCHMARK_MODELS: list[ModelName] = [
    "convnext_tiny",
    "swin_t",
    "efficientnet_v2_s",
    "efficientnet_b2",
]
