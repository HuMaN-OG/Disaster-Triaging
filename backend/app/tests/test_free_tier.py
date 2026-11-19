"""Free-tier model uses the actual labeled research dataset; no mocked ML scoring."""
import pytest
from app.services.lite_text_model import FreeTierTextPredictor, _terms, REPORTS_PATH
from app.services.ml_service import ProductionMultimodalPredictorAdapter
from app.core.config import settings
from fastapi import HTTPException


def test_free_tier_estimator_uses_labeled_dataset_and_scores_text():
    assert REPORTS_PATH.is_file()
    model = FreeTierTextPredictor()
    score, label, language = model.predict(
        "Flood waters are rising around the main street and people need evacuation help."
    )
    assert 0 <= score <= 100
    assert label in {"Very Low", "Low", "Medium", "High", "Critical"}
    assert language == "English"


def test_lite_model_rejects_unsupported_input_instead_of_inventing_scores():
    model = FreeTierTextPredictor()
    with pytest.raises(ValueError):
        model.predict("")
    with pytest.raises(HTTPException) as exc:
        model.predict("zzzzzzzzzx qqqqqqqqqx")
    assert exc.value.status_code == 503


def test_free_tier_adapter_is_text_only(monkeypatch):
    monkeypatch.setattr(settings, "LITE_MODEL_ENABLED", True)
    adapter = ProductionMultimodalPredictorAdapter()
    adapter.load_models()
    assert adapter.is_loaded
    assert adapter._predictor is None
    assert adapter._text_predictor is not None
    response = adapter.predict(text_report="Flood waters have entered several streets and buildings.")
    assert 0 <= response["priority_score"] <= 100
    assert response["confidence"] == 0.0
    assert response["disaster_type"] == "Unknown"
    assert response["class_probabilities"] == {}
    with pytest.raises(HTTPException) as exc:
        adapter.predict(image_input="fake-photo.jpg")
    assert exc.value.status_code == 503


def test_hindi_keyword_translation_is_explicit_not_trained_multilingual():
    terms = _terms("भूकंप और पानी")
    assert "earthquake" in terms
    assert "water" in terms
