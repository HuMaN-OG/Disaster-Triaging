"""
Inference API for AI Image Authenticity Detection.

Accepts an image path or PIL Image and returns:
- AI Generated / Real Photograph classification
- Confidence score
- Detailed model metadata
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from src.authenticity.config import AuthenticityConfig
from src.authenticity.model import AuthenticityClassifier
from src.authenticity.checkpoint import CheckpointManager
from src.authenticity.logger import get_logger

logger = get_logger("authenticity.predict")


class AuthenticityInference:
    """Production Inference class for AI Image Authenticity Classifier."""

    def __init__(self, config: AuthenticityConfig = None, model_path: str = None):
        self.config = config or AuthenticityConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Resize((self.config.model.img_size, self.config.model.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.model = AuthenticityClassifier(self.config.model)
        self.loaded_checkpoint_path = None
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, model_path: str = None):
        paths_to_try = []
        if model_path:
            paths_to_try.append(model_path)
        
        best_pth = os.path.join(self.config.models_dir, "best_model.pth")
        best_pt = os.path.join(self.config.models_dir, f"{self.config.model.architecture}_authenticity.pt")
        ckpt_best = os.path.join(self.config.checkpoints_dir, "best_model.pt")
        ckpt_latest = os.path.join(self.config.checkpoints_dir, "latest_checkpoint.pt")

        paths_to_try.extend([best_pth, best_pt, ckpt_best, ckpt_latest])

        # Also search checkpoints dir for epoch_*.pt if needed
        if os.path.exists(self.config.checkpoints_dir):
            epoch_files = sorted(
                [os.path.join(self.config.checkpoints_dir, f) for f in os.listdir(self.config.checkpoints_dir) if f.startswith("epoch_") and f.endswith(".pt")],
                reverse=True
            )
            paths_to_try.extend(epoch_files)

        loaded = False
        for path in paths_to_try:
            if path and os.path.exists(path):
                try:
                    state = torch.load(path, map_location=self.device)
                    if isinstance(state, dict) and "model_state_dict" in state:
                        self.model.load_state_dict(state["model_state_dict"])
                    elif isinstance(state, dict):
                        self.model.load_state_dict(state)
                    self.loaded_checkpoint_path = path
                    logger.info(f"Successfully loaded authenticity model weights from {path}")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Could not load weights from {path}: {e}")

        if not loaded:
            logger.warning("No trained weights found for AuthenticityInference! Initialized with random weights.")

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Run inference on input image.
        Returns dict containing:
        - classification: 'AI Generated' or 'Real Photograph'
        - confidence: float (0.0 to 1.0)
        - model_info: dict with architecture, weights path, thresholds, device
        """
        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            if not os.path.exists(img_path_str):
                raise FileNotFoundError(f"Image not found: {img_path_str}")
            image = Image.open(img_path_str).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            ai_prob = float(torch.sigmoid(logits).item())

        is_ai = ai_prob >= 0.5
        classification = "AI Generated" if is_ai else "Real Photograph"
        confidence = ai_prob if is_ai else (1.0 - ai_prob)

        # Assessment details
        if ai_prob < self.config.thresholds.pass_threshold:
            assessment = "PASS"
        elif ai_prob > self.config.thresholds.flagged_threshold:
            assessment = "FLAGGED"
        else:
            assessment = "UNCERTAIN"

        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "ai_probability": round(ai_prob, 4),
            "assessment": assessment,
            "is_ai_generated": is_ai,
            "model_info": {
                "architecture": self.config.model.architecture,
                "weights_path": self.loaded_checkpoint_path,
                "device": str(self.device),
                "thresholds": {
                    "pass_threshold": self.config.thresholds.pass_threshold,
                    "flagged_threshold": self.config.thresholds.flagged_threshold,
                }
            }
        }


# Module level singleton helper
_inference_instance = None


def predict_authenticity(image_input: Union[str, Path, Image.Image], model_path: str = None) -> Dict[str, Any]:
    """Helper function to run single-image authenticity prediction."""
    global _inference_instance
    if _inference_instance is None or model_path is not None:
        _inference_instance = AuthenticityInference(model_path=model_path)
    return _inference_instance.predict(image_input)
