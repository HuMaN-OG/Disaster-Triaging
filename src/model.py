"""
EfficientNet-B0 disaster image classifier.

Backbone starts frozen (classifier head only); call ``unfreeze_backbone()``
when ready for end-to-end fine-tuning.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.utils.image_classifier import NUM_CLASSES


class DisasterClassifier(nn.Module):
    """
    EfficientNet-B0 with a replaced 6-class classifier head.

    Parameters
    ----------
    num_classes:
        Number of output classes (default: 6).
    pretrained:
        Load ImageNet-pretrained EfficientNet-B0 weights.
    freeze_backbone:
        If True, freeze all feature-extractor parameters and train only the head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    # ------------------------------------------------------------------
    # Freeze / unfreeze API (ready for later fine-tuning)
    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze EfficientNet feature extractor; keep classifier trainable."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier")
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def trainable_parameter_counts(self) -> dict[str, int]:
        """Return counts of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def save(self, path: str | Any, extra: dict[str, Any] | None = None) -> None:
        """
        Persist model weights and optional metadata.

        Stored dict keys: ``model_state_dict``, ``num_classes``, ``backbone_frozen``,
        plus any keys from ``extra``.
        """
        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.backbone.classifier[1].out_features,
            "backbone_frozen": self._backbone_frozen,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Any,
        device: torch.device | str = "cpu",
        freeze_backbone: bool | None = None,
    ) -> tuple["DisasterClassifier", dict[str, Any]]:
        """
        Load a checkpoint produced by :meth:`save`.

        Returns the model and the full checkpoint dictionary.
        """
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # PyTorch < 2.0 does not accept weights_only
            checkpoint = torch.load(path, map_location=device)
        num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
        frozen = checkpoint.get("backbone_frozen", True)
        if freeze_backbone is not None:
            frozen = freeze_backbone

        model = cls(num_classes=num_classes, pretrained=False, freeze_backbone=frozen)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: torch.device | str | None = None,
) -> DisasterClassifier:
    """Factory helper that builds and moves the model to ``device``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisasterClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    return model.to(device)
