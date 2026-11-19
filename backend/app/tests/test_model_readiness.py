import pytest
from fastapi import HTTPException

from app.services.ml_service import ProductionMultimodalPredictorAdapter


def test_missing_models_never_fabricate_image_or_text_predictions():
    """The live adapter must return 503, not a confident fake flood prediction."""
    adapter = ProductionMultimodalPredictorAdapter()
    adapter.is_loaded = True
    adapter._predictor = None
    adapter._text_predictor = None

    with pytest.raises(HTTPException) as text_exc:
        adapter.predict(text_report="Flood water is rising around the buildings")
    assert text_exc.value.status_code == 503
    assert "not loaded" in text_exc.value.detail

    with pytest.raises(HTTPException) as image_exc:
        adapter.predict(image_input="/tmp/unavailable-checkpoint.jpg")
    assert image_exc.value.status_code == 503
    assert "Image classification is unavailable" in image_exc.value.detail


def test_text_model_only_still_reports_no_image_probabilities():
    """When only a trained text model exists, make its narrower scope explicit."""
    class RealTextModelStub:
        def predict(self, report):
            return 74.5, "High", "English"

    adapter = ProductionMultimodalPredictorAdapter()
    adapter.is_loaded = True
    adapter._text_predictor = RealTextModelStub()
    result = adapter.predict(text_report="Water is rising")
    assert result["priority_score"] == 74.5
    assert result["priority_level"] == "High"
    assert result["confidence"] == 0.0
    assert result["class_probabilities"] == {}
    assert result["disaster_type"] == "Unknown"

    with pytest.raises(HTTPException) as exc:
        adapter.predict(image_input="photo.jpg", text_report="Water is rising")
    assert exc.value.status_code == 503
