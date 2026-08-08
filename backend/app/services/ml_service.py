from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
from PIL import Image
import torch

from app.utils.language import detect_language


class BasePredictor(ABC):
    """
    Abstract Model-Agnostic Predictor Interface.
    All ML models (present or future) must conform to this interface.
    """

    @abstractmethod
    def predict(
        self,
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        text_report: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified prediction interface.
        Must return a dict matching the standard prediction schema:
        {
            "disaster_type": str,
            "priority_score": float,
            "priority_level": str,
            "confidence": float,
            "language_detected": str,
            "class_probabilities": Dict[str, float]
        }
        """
        pass


class ProductionMultimodalPredictorAdapter(BasePredictor):
    """
    Concrete adapter wrapping the production MultimodalPredictor from src/inference/predict.py.
    This module is the ONLY file in the backend that imports src/inference/predict.py.
    """

    def __init__(self):
        self._predictor = None
        self._authenticity_predictor = None
        self.is_loaded = False

    def load_models(self) -> None:
        """Load ML model checkpoints ONCE during app startup."""
        if not self.is_loaded:
            from src.inference.predict import MultimodalPredictor
            self._predictor = MultimodalPredictor(image_model_type="ensemble")
            try:
                from src.authenticity.predict import AuthenticityInference
                self._authenticity_predictor = AuthenticityInference()
            except Exception as e:
                import logging
                logging.getLogger("uvicorn.error").warning(f"Could not load AuthenticityInference: {e}")
            self.is_loaded = True

    def predict_authenticity(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        if not self.is_loaded or self._authenticity_predictor is None:
            from src.authenticity.predict import AuthenticityInference
            self._authenticity_predictor = AuthenticityInference()
            self.is_loaded = True
        return self._authenticity_predictor.predict(image_input)

    def predict(
        self,
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        text_report: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_loaded or self._predictor is None:
            self.load_models()

        has_image = image_input is not None
        has_text = text_report is not None and bool(text_report.strip())

        if not has_image and not has_text:
            raise ValueError("Prediction requires at least an image_input or text_report.")

        # Detect language if text provided
        lang = detect_language(text_report) if has_text else "English"
        authenticity_info = None

        if has_image:
            try:
                if self._authenticity_predictor is None:
                    from src.authenticity.predict import AuthenticityInference
                    self._authenticity_predictor = AuthenticityInference()
                authenticity_info = self._authenticity_predictor.predict(image_input)
            except Exception:
                authenticity_info = None

        if has_image and has_text:
            # Full multimodal inference
            res = self._predictor.predict(text_report=text_report, image_input=image_input)
            return {
                "disaster_type": res.disaster_type,
                "priority_score": round(float(res.priority_score), 1),
                "priority_level": res.priority_level,
                "confidence": round(float(res.confidence), 4),
                "language_detected": getattr(res, "language_detected", lang),
                "class_probabilities": {
                    k: round(float(v), 4) for k, v in res.class_probabilities.items()
                },
                "authenticity": authenticity_info,
            }

        elif has_image and not has_text:
            # Image only inference
            disaster_type, confidence, class_probabilities = self._predictor.image_predictor.predict(image_input)
            return {
                "disaster_type": disaster_type,
                "priority_score": 50.0,  # Default neutral score when text is absent
                "priority_level": "Medium",
                "confidence": round(float(confidence), 4),
                "language_detected": "English",
                "class_probabilities": {
                    k: round(float(v), 4) for k, v in class_probabilities.items()
                },
                "authenticity": authenticity_info,
            }

        else:  # has_text and not has_image
            # Text only priority scoring
            from src.predict_text import predict_priority_score
            text_res = predict_priority_score(
                text_report,
                self._predictor.text_predictor.model,
                self._predictor.text_predictor.encoder,
            )
            # Default uniform class probabilities when image is absent
            default_classes = ["Flood", "Earthquake", "Urban Fire", "Wild Fire", "Landslide", "Drought"]
            uniform_prob = round(1.0 / len(default_classes), 4)
            class_probs = {c: uniform_prob for c in default_classes}

            return {
                "disaster_type": "Unknown",
                "priority_score": round(float(text_res["priority_score"]), 1),
                "priority_level": text_res["ui_category"],
                "confidence": 0.5,
                "language_detected": lang,
                "class_probabilities": class_probs,
                "authenticity": None,
            }


class PredictionService:
    """
    Model-Agnostic Prediction Service.
    Interacts strictly via the BasePredictor interface.
    """

    def __init__(self, predictor: BasePredictor):
        self.predictor = predictor

    def run_prediction(
        self,
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        text_report: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.predictor.predict(image_input=image_input, text_report=text_report)

    def run_authenticity_prediction(
        self,
        image_input: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        if hasattr(self.predictor, "predict_authenticity"):
            return self.predictor.predict_authenticity(image_input)
        from src.authenticity.predict import predict_authenticity
        return predict_authenticity(image_input)


# Global singleton instance for Dependency Injection
_predictor_adapter = ProductionMultimodalPredictorAdapter()


def initialize_ml_models() -> None:
    """Explicit startup hook called by FastAPI lifespan."""
    _predictor_adapter.load_models()


def get_prediction_service() -> PredictionService:
    """Dependency Provider for PredictionService."""
    return PredictionService(predictor=_predictor_adapter)
