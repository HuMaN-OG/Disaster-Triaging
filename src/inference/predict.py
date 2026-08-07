"""
Inference module for disaster image classification, text priority scoring, and multimodal rescue triage.

Provides `ImagePredictor` for disaster type prediction from photos,
and `MultimodalPredictor` for combining citizen reports (text) + uploaded photos (image).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.benchmark_models import BenchmarkClassifier, ModelName
from src.dataset import get_val_transforms
from src.utils.image_classifier import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    MODELS_DIR,
)


@dataclass
class ImagePredictionResult:
    """Dataclass holding image disaster prediction output."""

    image_path: str
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: dict[str, float]


@dataclass
class MultimodalPredictionResult:
    """Dataclass holding combined image + text rescue triage prediction."""

    disaster_type: str
    image_confidence: float
    priority_score: float
    priority_class: str
    image_class_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Format prediction as a JSON-serializable dictionary."""
        return {
            "disaster_type": self.disaster_type,
            "priority": self.priority_class,
            "severity_score": round(self.priority_score, 1),
            "confidence": round(self.image_confidence, 4),
            "class_probabilities": {
                k: round(v, 4) for k, v in self.image_class_probabilities.items()
            },
        }


class ImagePredictor:
    """
    Inference predictor for disaster image classification.

    Parameters
    ----------
    model_type:
        Architecture to use for inference (``"swin_t"``, ``"convnext_tiny"``,
        ``"vit_b_16"``, ``"efficientnet_v2_s"``, or ``"ensemble"``).
    device:
        Torch device to execute inference on.
    """

    def __init__(
        self,
        model_type: str = "swin_t",
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_type = model_type
        self.transform = get_val_transforms()
        self.models: dict[str, BenchmarkClassifier] = {}
        self.weights: dict[str, float] = {}

        if model_type == "ensemble":
            ensemble_weights = {
                "swin_t": 0.30,
                "convnext_tiny": 0.20,
                "vit_b_16": 0.20,
                "efficientnet_v2_s": 0.30,
            }
            for name, w in ensemble_weights.items():
                ckpt = MODELS_DIR / f"disaster_classifier_{name}.pth"
                if ckpt.exists():
                    m, _ = BenchmarkClassifier.load(ckpt, device=self.device, finetune_mode="frozen")
                    m.eval()
                    self.models[name] = m
                    self.weights[name] = w
            if not self.models:
                raise FileNotFoundError("No trained benchmark model checkpoints found in models/")
        else:
            ckpt = MODELS_DIR / f"disaster_classifier_{model_type}.pth"
            if not ckpt.exists():
                ckpt = MODELS_DIR / "disaster_classifier.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint for {model_type} not found at {ckpt}")
            m, _ = BenchmarkClassifier.load(ckpt, device=self.device, finetune_mode="frozen")
            m.eval()
            self.models[model_type] = m
            self.weights[model_type] = 1.0

    def predict(self, image_input: str | Path | Image.Image) -> ImagePredictionResult:
        """
        Predict disaster class for a single image file or PIL Image object.
        """
        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else "PIL.Image"

        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                pil_img = img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            weighted_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            total_w = sum(self.weights.values())

            for name, model in self.models.items():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                weighted_probs += (self.weights[name] / total_w) * probs

        top_idx = int(np.argmax(weighted_probs))
        confidence = float(weighted_probs[top_idx])
        class_probs = {
            CLASS_NAMES[i]: float(weighted_probs[i]) for i in range(len(CLASS_NAMES))
        }

        return ImagePredictionResult(
            image_path=image_path_str,
            predicted_class=IDX_TO_CLASS[top_idx],
            predicted_class_idx=top_idx,
            confidence=confidence,
            class_probabilities=class_probs,
        )

    def predict_batch(
        self, image_inputs: Sequence[str | Path | Image.Image]
    ) -> list[ImagePredictionResult]:
        """Predict disaster classes for a batch of images."""
        return [self.predict(img) for img in image_inputs]


class MultimodalPredictor:
    """
    End-to-end Multimodal Rescue Triage Predictor.

    Fuses predictions from the Image Classifier (Disaster Type) and
    Text Classifier (Priority Score / Severity Class).
    """

    def __init__(
        self,
        image_model_type: str = "ensemble",
        device: torch.device | str | None = None,
    ) -> None:
        from src.predict_text import load_pipeline
        from src.text_dataset import MODEL_META_PATH, XGBOOST_PATH

        self.image_predictor = ImagePredictor(model_type=image_model_type, device=device)
        self.text_model, self.text_encoder = load_pipeline(XGBOOST_PATH, MODEL_META_PATH)

    def predict(
        self,
        text_report: str,
        image_input: str | Path | Image.Image,
    ) -> MultimodalPredictionResult:
        """
        Given citizen report text and an image, produce a combined MultimodalPredictionResult.
        """
        from src.predict_text import predict_priority_score

        # 1. Image prediction
        img_res = self.image_predictor.predict(image_input)

        # 2. Text prediction
        text_res = predict_priority_score(text_report, self.text_model, self.text_encoder)

        return MultimodalPredictionResult(
            disaster_type=img_res.predicted_class,
            image_confidence=img_res.confidence,
            priority_score=text_res["priority_score"],
            priority_class=text_res["ui_category"],
            image_class_probabilities=img_res.class_probabilities,
        )


def predict_image(image_path: str | Path, model_type: str = "swin_t") -> ImagePredictionResult:
    """Convenience helper for quick single-image inference."""
    predictor = ImagePredictor(model_type=model_type)
    return predictor.predict(image_path)
