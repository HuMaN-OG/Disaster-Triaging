import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Union
from .config import AuthenticityConfig
from .model import AuthenticityClassifier
from .checkpoint import CheckpointManager
from .logger import get_logger

logger = get_logger("authenticity.predictor")

class AuthenticityPredictor:
    def __init__(self, config: AuthenticityConfig = None, checkpoint_path: str = None, model: nn.Module = None, device: torch.device = None):
        self.config = config or AuthenticityConfig()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        self.transform = transforms.Compose([
            transforms.Resize((self.config.model.img_size, self.config.model.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        else:
            self.model = AuthenticityClassifier(self.config.model)
            ckpt_mgr = CheckpointManager(self.config.checkpoints_dir)
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=self.device)
            else:
                ckpt = ckpt_mgr.load_best(self.device)
                
            if ckpt and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded model weights successfully into AuthenticityPredictor.")
            else:
                logger.warning("No checkpoint found! Predictor initialized with un-trained / random weights.")
            self.model.to(self.device)
            self.model.eval()

    def predict_image(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            image = Image.open(image_input).convert("RGB")
            image_path = image_input
        else:
            image = image_input.convert("RGB")
            image_path = "PIL.Image"

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            ai_prob = float(torch.sigmoid(logits).item())
            real_prob = 1.0 - ai_prob

        # Determine Assessment Status based on Configured Decision Thresholds
        if ai_prob < self.config.thresholds.pass_threshold:
            status = "PASS"
            explanation = "Authentic Real Photograph. Low probability of AI generation."
        elif ai_prob > self.config.thresholds.flagged_threshold:
            status = "FLAGGED"
            explanation = "Potential AI-Generated synthetic image. High probability of AI artifacts."
        else:
            status = "UNCERTAIN"
            explanation = "Ambiguous authenticity score. Requires manual review."

        return {
            "assessment": status,
            "ai_probability": round(ai_prob, 4),
            "real_probability": round(real_prob, 4),
            "thresholds": {
                "pass_threshold": self.config.thresholds.pass_threshold,
                "flagged_threshold": self.config.thresholds.flagged_threshold
            },
            "explanation": explanation,
            "image_path": image_path,
            "model_architecture": self.config.model.architecture
        }
