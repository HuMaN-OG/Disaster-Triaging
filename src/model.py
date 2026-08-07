"""
EfficientNet-B0 disaster image classifier.

Supports head-only training (frozen backbone), partial fine-tuning of the
last N feature blocks, or full end-to-end fine-tuning.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.utils.image_classifier import NUM_CLASSES, UNFREEZE_LAST_N_BLOCKS

FinetuneMode = Literal["frozen", "partial", "full"]


class DisasterClassifier(nn.Module):
    """
    EfficientNet-B0 with a replaced 5-class classifier head.

    Parameters
    ----------
    num_classes:
        Number of output classes (default: 5).
    pretrained:
        Load ImageNet-pretrained EfficientNet-B0 weights.
    finetune_mode:
        ``"frozen"`` — train classifier head only;
        ``"partial"`` — unfreeze last N feature blocks + head;
        ``"full"`` — train all parameters.
    unfreeze_last_n:
        Number of trailing ``features`` blocks to unfreeze in partial mode.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        finetune_mode: FinetuneMode = "partial",
        unfreeze_last_n: int = UNFREEZE_LAST_N_BLOCKS,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self._finetune_mode: FinetuneMode = "full"
        self._unfreeze_last_n = unfreeze_last_n
        self._apply_finetune_mode(finetune_mode)

    # ------------------------------------------------------------------
    # Freeze / unfreeze API
    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze EfficientNet feature extractor; keep classifier trainable."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier")
        self._finetune_mode = "frozen"

    def partial_finetune(self, unfreeze_last_n: int | None = None) -> None:
        """Unfreeze the last N feature blocks and the classifier head."""
        n = unfreeze_last_n if unfreeze_last_n is not None else self._unfreeze_last_n
        self._unfreeze_last_n = n

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        feature_modules = list(self.backbone.features.children())
        for module in feature_modules[-n:]:
            for param in module.parameters():
                param.requires_grad = True

        self._finetune_mode = "partial"

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
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
        """True when only the classifier head is trainable."""
        return self._finetune_mode == "frozen"

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        """Trainable backbone parameters excluding the classifier head."""
        return [
            param
            for name, param in self.backbone.named_parameters()
            if param.requires_grad and not name.startswith("classifier")
        ]

    def classifier_trainable_parameters(self) -> list[nn.Parameter]:
        """Trainable classifier-head parameters."""
        return [p for p in self.backbone.classifier.parameters() if p.requires_grad]

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

        Stored dict keys: ``model_state_dict``, ``num_classes``, ``finetune_mode``,
        plus any keys from ``extra``.
        """
        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.backbone.classifier[1].out_features,
            "finetune_mode": self._finetune_mode,
            "backbone_frozen": self.backbone_frozen,
            "model_type": "efficientnet_b0",
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
        """
        Load a checkpoint produced by :meth:`save`.

        Returns the model and the full checkpoint dictionary.
        """
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
        mode: FinetuneMode = checkpoint.get("finetune_mode", "partial")
        if finetune_mode is not None:
            mode = finetune_mode

        model = cls(num_classes=num_classes, pretrained=False, finetune_mode=mode)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    finetune_mode: FinetuneMode = "partial",
    unfreeze_last_n: int = UNFREEZE_LAST_N_BLOCKS,
    device: torch.device | str | None = None,
) -> DisasterClassifier:
    """Factory helper that builds and moves the model to ``device``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisasterClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        finetune_mode=finetune_mode,
        unfreeze_last_n=unfreeze_last_n,
    )
    return model.to(device)


# Backward-compatible alias for callers expecting freeze_backbone kwarg
def build_model_legacy(
    freeze_backbone: bool = True,
    device: torch.device | str | None = None,
    **kwargs: Any,
) -> DisasterClassifier:
    mode: FinetuneMode = "frozen" if freeze_backbone else "full"
    return build_model(finetune_mode=mode, device=device, **kwargs)
