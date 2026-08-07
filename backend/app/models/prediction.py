from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    disaster_type = Column(String, nullable=False)
    priority_score = Column(Float, nullable=False)
    priority_level = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=False)  # JSON dict with all 6 classes
    language_detected = Column(String, nullable=False, default="English")
    model_version = Column(String, nullable=False, default="Multimodal-6Class-v1")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    incident = relationship("Incident", back_populates="predictions")
