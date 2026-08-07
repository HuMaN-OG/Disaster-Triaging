"""
Inference engine for Image Authenticity Verification.

Returns configurable status ("PASS", "FLAGGED", "UNCERTAIN") alongside
both real_probability and ai_probability. Preserves generator metadata if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from image_model.configs import AuthenticityConfig, AuthenticityVerificationConfig, DEFAULT_CLASS_NAMES, MODELS_DIR
from image_model.datasets import get_authenticity_val_transforms
from image_model.models import DisasterClassifier


@dataclass
class AuthenticityVerificationResult:
    """
    Standard output structure for Image Authenticity Verification.
    
    Exposes:
    - status: "PASS", "FLAGGED", "UNCERTAIN"
    - real_probability: float (0.0 to 1.0)
    - ai_probability: float (0.0 to 1.0)
    - is_real: bool
    - predicted_class: str
    - confidence: float
    - generator_source: str (e.g., "Midjourney", "DALL-E", "Unknown")
    """

    image_path: str
    status: str                       # "PASS" | "FLAGGED" | "UNCERTAIN"
    real_probability: float
    ai_probability: float
    is_real: bool
    predicted_class: str
    confidence: float
    generator_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "status": self.status,
            "real_probability": round(self.real_probability, 4),
            "ai_probability": round(self.ai_probability, 4),
            "is_real": self.is_real,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "generator_source": self.generator_source,
        }


class ImageAuthenticityVerifier:
    """
    Inference Engine for Image Authenticity Verification.

    Parameters
    ----------
    checkpoint_path:
        Path to trained PyTorch model weights.
    verification_config:
        Custom status threshold configuration. If None, uses default settings.
    device:
        Torch execution device.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        verification_config: AuthenticityVerificationConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if verification_config is None:
            self.verification_config = AuthenticityVerificationConfig()
        else:
            self.verification_config = verification_config

        if checkpoint_path is None:
            checkpoint_path = MODELS_DIR / "authenticity_best_model.pth"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Authenticity model checkpoint not found at {checkpoint_path}")

        self.model, self.checkpoint = DisasterClassifier.load(
            checkpoint_path, device=self.device, finetune_mode="frozen"
        )
        self.model.eval()

        self.class_names = self.checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
        self.transform = get_authenticity_val_transforms()

    def verify(
        self,
        image_input: str | Path | Image.Image,
        generator_source: str = "unknown",
    ) -> AuthenticityVerificationResult:
        """
        Verify the authenticity of an image input.
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
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Assuming binary: Class 0 = Real, Class 1 = AI Generated
        if len(probs) >= 2:
            real_prob = float(probs[0])
            ai_prob = float(probs[1])
        else:
            real_prob = float(probs[0])
            ai_prob = 1.0 - real_prob

        # Determine verification status based on configurable thresholds
        cfg = self.verification_config
        if ai_prob >= cfg.flagged_threshold:
            status = "FLAGGED"
            is_real = False
        elif ai_prob <= cfg.pass_threshold:
            status = "PASS"
            is_real = True
        else:
            status = "UNCERTAIN"
            is_real = real_prob > ai_prob

        top_idx = int(np.argmax(probs))
        predicted_class = self.class_names[top_idx] if top_idx < len(self.class_names) else "Unknown"
        confidence = float(probs[top_idx])

        return AuthenticityVerificationResult(
            image_path=image_path_str,
            status=status,
            real_probability=real_prob,
            ai_probability=ai_prob,
            is_real=is_real,
            predicted_class=predicted_class,
            confidence=confidence,
            generator_source=generator_source,
        )

    def verify_batch(
        self, image_inputs: Sequence[str | Path | Image.Image]
    ) -> list[AuthenticityVerificationResult]:
        """Run batch verification across multiple images."""
        return [self.verify(img) for img in image_inputs]


# Backward-compatible alias
ImagePredictor = ImageAuthenticityVerifier
