from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
from PIL import Image
import logging
from fastapi import HTTPException, status

from app.utils.language import detect_language
from app.core.config import settings

logger = logging.getLogger(__name__)


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
        self._text_predictor = None
        self.is_loaded = False

    def load_models(self) -> None:
        """Load the real trained models only. Never substitute a fabricated classification."""
        if self.is_loaded:
            return
        if settings.LITE_MODEL_ENABLED:
            from app.services.lite_text_model import FreeTierTextPredictor
            try:
                self._text_predictor = FreeTierTextPredictor()
                logger.info("Loaded free-tier text-only estimator from labeled research dataset")
            except Exception:
                logger.exception("Failed to initialize free-tier text estimator")
                self._text_predictor = None
            self._predictor = None
            self.is_loaded = True
            return
        try:
            from src.inference.predict import MultimodalPredictor
            self._predictor = MultimodalPredictor(image_model_type="ensemble")
        except Exception:
            # This repository does not contain the trained vision .pth checkpoints.
            # A valid text checkpoint can still support honest text-only predictions.
            logger.warning("Full multimodal model unavailable; trying text-only model", exc_info=True)
            self._predictor = None
            try:
                from src.inference.predict import TextPredictor
                self._text_predictor = TextPredictor()
            except Exception:
                logger.exception("Text-only prediction model unavailable")
                self._text_predictor = None
        finally:
            self.is_loaded = True

    def predict(
        self,
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        text_report: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()

        if self._predictor is not None:
            has_image = image_input is not None
            has_text = text_report is not None and bool(text_report.strip())

            if not has_image and not has_text:
                raise ValueError("Prediction requires at least an image_input or text_report.")

            lang = detect_language(text_report) if has_text else "English"

            if has_image and has_text:
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
                }
            elif has_image and not has_text:
                disaster_type, confidence, class_probabilities = self._predictor.image_predictor.predict(image_input)
                return {
                    "disaster_type": disaster_type,
                    "priority_score": 50.0,
                    "priority_level": "Medium",
                    "confidence": round(float(confidence), 4),
                    "language_detected": "English",
                    "class_probabilities": {
                        k: round(float(v), 4) for k, v in class_probabilities.items()
                    },
                }
            else:
                from src.predict_text import predict_priority_score
                text_res = predict_priority_score(
                    text_report,
                    self._predictor.text_predictor.model,
                    self._predictor.text_predictor.encoder,
                )
                # Text-only regression does not calculate a vision class distribution.
                class_probs = {}
                return {
                    "disaster_type": "Unknown",
                    "priority_score": round(float(text_res["priority_score"]), 1),
                    "priority_level": text_res["ui_category"],
                    "confidence": 0.0,  # No image-classification confidence is available.
                    "language_detected": lang,
                    "class_probabilities": class_probs,
                }

        # A missing vision model must never be disguised as a confident Flood classification.
        # The browser can still submit text-only reports if the trained text model exists.
        has_image = image_input is not None
        has_text = bool(text_report and text_report.strip())

        if has_image:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Image classification is unavailable: trained vision checkpoints are not installed. Please submit a text-only report.",
            )
        if not has_text:
            raise ValueError("Prediction requires a text report or an available image model.")
        if self._text_predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Text prediction model is not loaded. Check the backend model assets and inference dependencies.",
            )

        score, level, lang = self._text_predictor.predict(text_report)
        return {
            "disaster_type": "Unknown",  # Text regression does not classify image categories.
            "priority_score": round(float(score), 1),
            "priority_level": level,
            "confidence": 0.0,
            "language_detected": lang,
            "class_probabilities": {},
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


# Global singleton instance for Dependency Injection
_predictor_adapter = ProductionMultimodalPredictorAdapter()


def initialize_ml_models() -> None:
    """Explicit startup hook called by FastAPI lifespan."""
    _predictor_adapter.load_models()


def get_prediction_service() -> PredictionService:
    """Dependency Provider for PredictionService."""
    return PredictionService(predictor=_predictor_adapter)
