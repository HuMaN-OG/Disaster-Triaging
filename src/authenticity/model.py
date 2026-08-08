import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import timm
from .config import ModelConfig
from .logger import get_logger

logger = get_logger("authenticity.model")

class AuthenticityClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.architecture = config.architecture.lower()
        
        logger.info(f"Initializing AuthenticityClassifier with backbone: {self.architecture}, pretrained={config.pretrained}")

        if self.architecture in ["convnext_tiny", "convnext"]:
            try:
                # Try torchvision first
                weights = torchvision_models.ConvNeXt_Tiny_Weights.DEFAULT if config.pretrained else None
                backbone = torchvision_models.convnext_tiny(weights=weights)
                in_features = backbone.classifier[2].in_features
                backbone.classifier[2] = nn.Identity()
                self.backbone = backbone
                self.classifier = nn.Sequential(
                    nn.Dropout(p=config.dropout_rate),
                    nn.Linear(in_features, config.num_classes)
                )
            except Exception as e:
                logger.info(f"Torchvision convnext_tiny failed ({e}), falling back to timm.")
                self.backbone = timm.create_model("convnext_tiny", pretrained=config.pretrained, num_classes=config.num_classes)
                self.classifier = None
                
        elif self.architecture in ["swin_t", "swin_tiny", "swin"]:
            try:
                weights = torchvision_models.Swin_T_Weights.DEFAULT if config.pretrained else None
                backbone = torchvision_models.swin_t(weights=weights)
                in_features = backbone.head.in_features
                backbone.head = nn.Identity()
                self.backbone = backbone
                self.classifier = nn.Sequential(
                    nn.Dropout(p=config.dropout_rate),
                    nn.Linear(in_features, config.num_classes)
                )
            except Exception as e:
                logger.info(f"Torchvision swin_t failed ({e}), falling back to timm.")
                self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=config.pretrained, num_classes=config.num_classes)
                self.classifier = None
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}. Choose convnext_tiny or swin_t")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is not None:
            features = self.backbone(x)
            logits = self.classifier(features)
        else:
            logits = self.backbone(x)
            
        return logits.squeeze(-1) if logits.ndim > 1 and logits.shape[-1] == 1 else logits

    def get_probability(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
