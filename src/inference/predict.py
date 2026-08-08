"""
Production Inference API for Disaster Image Classification, Text Priority Scoring,
and Multimodal Rescue Triage.

Provides:
- `ImagePredictor`: Predicts disaster type from citizen photos.
- `TextPredictor`: Predicts priority score (0–100) and detects language (English, Hindi, Hinglish).
- `MultimodalPredictor`: Single unified production API supporting Image-only, Text-only,
  and Multimodal (Image + Text) rescue triage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.benchmark_models import BenchmarkClassifier
from src.dataset import get_val_transforms
from src.hinglish_normalizer import normalize_hinglish_text
from src.text_dataset import (
    CATBOOST_PATH,
    MODEL_META_PATH,
    MODELS_DIR as TEXT_MODELS_DIR,
    clip_score,
    score_to_ui_category,
)
from src.utils.image_classifier import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    MODELS_DIR as IMAGE_MODELS_DIR,
)


def detect_language(text: str) -> str:
    """
    Detect language of a citizen report: 'Hindi', 'Hinglish', or 'English'.
    """
    if not text or not text.strip():
        return "N/A"

    # Devanagari script check
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"

    # Hinglish key token heuristic
    hinglish_keywords = {
        "hai", "bohot", "samajh", "gaya", "gaye", "raha", "rahi", "rahe",
        "kya", "paani", "aag", "rasta", "phasa", "phanse", "bhejo", "ghar",
        "keechad", "chattano", "baadh", "dhua", "ghayal", "log", "chahiye",
        "levee", "makan", "jao", "madad", "bache", "batao", "kaise", "yahan",
        "wahan", "subah", "raat", "sab", "apne", "mere", "se", "ko", "par",
        "bhi", "thi", "tha", "hue", "ho", "kar", "nahin", "nahi"
    }
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    matches = tokens.intersection(hinglish_keywords)
    if len(matches) >= 2 or any(w in tokens for w in ["hinglish", "paani", "keechad", "phanse", "phasa", "aag", "bhejo", "chahiye"]):
        return "Hinglish"

    return "English"


@dataclass
class PredictionResult:
    """
    Unified production prediction result for hackathon submission & deployment.
    """

    disaster_type: str
    priority_score: float
    priority_level: str
    confidence: float
    language_detected: str
    input_mode: str
    class_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Format prediction as a clean JSON-serializable dictionary."""
        return {
            "disaster_type": self.disaster_type,
            "priority_score": round(self.priority_score, 1),
            "priority_level": self.priority_level,
            "confidence": round(self.confidence, 4),
            "language_detected": self.language_detected,
            "input_mode": self.input_mode,
            "class_probabilities": {
                k: round(v, 4) for k, v in self.class_probabilities.items()
            },
        }


class ImagePredictor:
    """Inference predictor for disaster image classification."""

    def __init__(
        self,
        model_type: str = "ensemble",
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
        self.class_names: list[str] = CLASS_NAMES
        first_ckpt_info = None

        if model_type == "ensemble":
            ensemble_weights = {
                "swin_t": 0.30,
                "convnext_tiny": 0.20,
                "vit_b_16": 0.20,
                "efficientnet_v2_s": 0.30,
            }
            for name, w in ensemble_weights.items():
                ckpt = IMAGE_MODELS_DIR / f"disaster_classifier_{name}.pth"
                if not ckpt.exists():
                    ckpt = IMAGE_MODELS_DIR / "disaster_classifier.pth"
                if ckpt.exists():
                    m, meta = BenchmarkClassifier.load(ckpt, device=self.device, finetune_mode="frozen")
                    m.eval()
                    self.models[name] = m
                    self.weights[name] = w
                    if first_ckpt_info is None and "class_names" in meta:
                        first_ckpt_info = meta["class_names"]
            if not self.models:
                ckpt = IMAGE_MODELS_DIR / "disaster_classifier.pth"
                if ckpt.exists():
                    m, meta = BenchmarkClassifier.load(ckpt, device=self.device, finetune_mode="frozen")
                    m.eval()
                    self.models["default"] = m
                    self.weights["default"] = 1.0
                    if "class_names" in meta:
                        first_ckpt_info = meta["class_names"]
                else:
                    raise FileNotFoundError(f"No trained image checkpoint found in {IMAGE_MODELS_DIR}")
        else:
            ckpt = IMAGE_MODELS_DIR / f"disaster_classifier_{model_type}.pth"
            if not ckpt.exists():
                ckpt = IMAGE_MODELS_DIR / "disaster_classifier.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint for {model_type} not found at {ckpt}")
            m, meta = BenchmarkClassifier.load(ckpt, device=self.device, finetune_mode="frozen")
            m.eval()
            self.models[model_type] = m
            self.weights[model_type] = 1.0
            if "class_names" in meta:
                first_ckpt_info = meta["class_names"]

        if first_ckpt_info:
            self.class_names = first_ckpt_info

    def predict(self, image_input: str | Path | Image.Image) -> tuple[str, float, dict[str, float]]:
        """Predict disaster class, confidence, and class probabilities."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                pil_img = img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        num_classes = len(self.class_names)
        with torch.no_grad():
            weighted_probs = np.zeros(num_classes, dtype=np.float32)
            total_w = sum(self.weights.values())

            for name, model in self.models.items():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                weighted_probs += (self.weights[name] / total_w) * probs

        top_idx = int(np.argmax(weighted_probs))
        confidence = float(weighted_probs[top_idx])
        class_probs = {
            self.class_names[i]: float(weighted_probs[i]) for i in range(num_classes)
        }
        return self.class_names[top_idx], confidence, class_probs


def check_life_safety_override(text: str) -> bool:
    text_lower = text.lower()
    safe_keywords = [
        "safe", "control", "no one trapped", "no injury", "no injuries", 
        "no help needed", "no rescue required", "koi ghayal nahi", "nahi chahiye",
        "safe hain", "safe hai", "theek hai"
    ]
    if any(s in text_lower for s in safe_keywords):
        return False
    emergency_keywords = [
        "trapped", "stranded",
        "faas gya", "faas gaya", "phas gaya", "phas gya", "fas gaya", "fas gya",
        "faas gye", "phas gye", "fas gye",
        "phansa", "phanse", "phasa", "fasa", "fase", "fanse", "faase", "phans",
        "atka", "atke"
    ]
    return any(w in text_lower for w in emergency_keywords)


class TextPredictor:
    """Inference predictor for multilingual text priority regression."""

    def __init__(self, model_path: Path | None = None) -> None:
        import joblib
        from catboost import CatBoostRegressor
        from sentence_transformers import SentenceTransformer

        ckpt = model_path if model_path is not None else CATBOOST_PATH
        if not ckpt.exists():
            ckpt = TEXT_MODELS_DIR / "catboost_multilingual_hinglish_augmented.cbm"
        if not ckpt.exists():
            raise FileNotFoundError(f"Text regressor model not found at {ckpt}")

        self.model = CatBoostRegressor()
        self.model.load_model(str(ckpt))

        sentence_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        if MODEL_META_PATH.exists():
            meta = joblib.load(MODEL_META_PATH)
            sentence_model_name = meta.get("sentence_model", sentence_model_name)

        self.encoder = SentenceTransformer(sentence_model_name)

    def predict(self, text_report: str) -> tuple[float, str, str]:
        """Predict priority score, UI category, and detected language."""
        lang = detect_language(text_report)
        norm_text = normalize_hinglish_text(text_report.strip())
        emb = self.encoder.encode([norm_text], convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        raw_score = float(self.model.predict(emb)[0])
        score = clip_score(raw_score)
        
        if check_life_safety_override(text_report):
            score = 95.0
            
        level = score_to_ui_category(score)
        return score, level, lang


class MultimodalPredictor:
    """
    Unified Production Predictor handling Image-only, Text-only, and Multimodal requests.
    """

    def __init__(
        self,
        image_model_type: str = "ensemble",
        device: torch.device | str | None = None,
    ) -> None:
        self.image_predictor = ImagePredictor(model_type=image_model_type, device=device)
        self.text_predictor = TextPredictor()

    def predict(
        self,
        image_input: str | Path | Image.Image | None = None,
        text_report: str | None = None,
    ) -> PredictionResult:
        """
        Execute prediction gracefully handling Image-only, Text-only, or Both.
        """
        has_image = image_input is not None
        has_text = text_report is not None and bool(text_report.strip())

        if not has_image and not has_text:
            raise ValueError("At least one input (image_input or text_report) must be provided.")

        if has_image and has_text:
            input_mode = "Multimodal (Image + Text)"
            disaster_type, img_conf, class_probs = self.image_predictor.predict(image_input)
            score, level, lang = self.text_predictor.predict(text_report)
            return PredictionResult(
                disaster_type=disaster_type,
                priority_score=score,
                priority_level=level,
                confidence=img_conf,
                language_detected=lang,
                input_mode=input_mode,
                class_probabilities=class_probs,
            )
        elif has_image:
            input_mode = "Image Only"
            disaster_type, img_conf, class_probs = self.image_predictor.predict(image_input)
            default_scores = {
                "Urban Fire": 85.0,
                "Wild Fire": 85.0,
                "Flood": 75.0,
                "Landslide": 70.0,
                "Earthquake": 80.0,
                "Drought": 50.0,
            }
            score = default_scores.get(disaster_type, 60.0)
            level = score_to_ui_category(score)
            return PredictionResult(
                disaster_type=disaster_type,
                priority_score=score,
                priority_level=level,
                confidence=img_conf,
                language_detected="N/A",
                input_mode=input_mode,
                class_probabilities=class_probs,
            )
        else:
            input_mode = "Text Only"
            score, level, lang = self.text_predictor.predict(text_report)
            text_lower = text_report.lower()
            if any(w in text_lower for w in ["fire", "aag", "smoke", "dhua"]):
                inferred_type = "Urban Fire / Wildfire"
            elif any(w in text_lower for w in ["flood", "water", "paani", "baadh", "levee"]):
                inferred_type = "Flood"
            elif any(w in text_lower for w in ["landslide", "mudflow", "rock", "keechad", "pathar"]):
                inferred_type = "Landslide"
            elif any(w in text_lower for w in ["earthquake", "building", "collapse", "rubble"]):
                inferred_type = "Earthquake"
            else:
                inferred_type = "Citizen Emergency Report"

            dummy_probs = {c: (1.0 if c in inferred_type else 0.0) for c in CLASS_NAMES}
            return PredictionResult(
                disaster_type=inferred_type,
                priority_score=score,
                priority_level=level,
                confidence=0.95,
                language_detected=lang,
                input_mode=input_mode,
                class_probabilities=dummy_probs,
            )
