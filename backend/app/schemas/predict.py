from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text_report: Optional[str] = Field(None, description="Citizen report text description")


class PredictionResultSchema(BaseModel):
    disaster_type: str
    priority_score: float
    priority_level: str
    confidence: float
    language_detected: str
    class_probabilities: Dict[str, float]


class PredictionAPIResponse(BaseModel):
    incident_id: Optional[int] = None
    disaster_type: str
    priority_score: float
    priority_level: str
    confidence: float
    language_detected: str
    class_probabilities: Dict[str, float]
    created_at: datetime
