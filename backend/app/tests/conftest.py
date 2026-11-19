"""Isolated backend API tests. Fake predictions NEVER run in production.

These fixtures validate incident persistence, authentication and API contract without
downloading multi-GB model weights or inventing live inference results.
Actual inference is checked separately through model-availability tests and deployment checks.
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.core.database import Base, get_db
from app.services.auth_service import seed_initial_users_and_teams
from app.services import incident_service as incident_service_module
from app.services.ml_service import (
    BasePredictor, PredictionService, get_prediction_service, _predictor_adapter,
)
from app.utils.language import detect_language
from app.core.config import settings

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_disaster.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


class _FixturePredictor(BasePredictor):
    """Only used by tests: deterministic fixture responses, not an ML model."""

    def predict(self, image_input=None, text_report=None):
        if image_input is None and not (text_report and text_report.strip()):
            raise ValueError("Text report or image required")
        text = (text_report or "").lower()
        safe = any(word in text for word in ("safe hain", "koi ghayal nahi", "no one trapped"))
        urgent = any(word in text for word in ("trapped", "stranded", "faas", "fas", "phas", "atke"))
        score = 95.0 if urgent and not safe else 50.0
        image_labels = ["Flood", "Earthquake", "Urban Fire", "Wild Fire", "Landslide", "Drought"]
        class_probs = {"Flood": 0.85, **{name: 0.03 for name in image_labels[1:]}} if image_input else {
            name: 1 / len(image_labels) for name in image_labels
        }
        return {
            "disaster_type": "Flood" if image_input else "Unknown",
            "priority_score": score,
            "priority_level": "Critical" if score == 95.0 else "Medium",
            "confidence": 0.85 if image_input else 0.0,
            "language_detected": detect_language(text_report or ""),
            "class_probabilities": class_probs,
        }


@pytest.fixture(scope="session", autouse=True)
def setup_database_and_models():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    settings.DEMO_ACCOUNTS_ENABLED = True
    settings.DEMO_ADMIN_PASSWORD = "Admin@123"
    settings.DEMO_RESPONDER_PASSWORD = "Responder@123"
    db = TestingSessionLocal()
    seed_initial_users_and_teams(db)
    db.close()

    fake_service = PredictionService(_FixturePredictor())
    previous = incident_service_module.get_prediction_service
    old_loaded = _predictor_adapter.is_loaded

    # Prevent the TestClient lifespan from trying to download real model weights.
    _predictor_adapter.is_loaded = True
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_prediction_service] = lambda: fake_service
    incident_service_module.get_prediction_service = lambda: fake_service
    try:
        yield
    finally:
        incident_service_module.get_prediction_service = previous
        _predictor_adapter.is_loaded = old_loaded
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
