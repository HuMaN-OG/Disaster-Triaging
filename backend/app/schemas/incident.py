from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, ConfigDict


class LocationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    address_text: Optional[str] = None


class IncidentCreate(BaseModel):
    title: Optional[str] = Field(None, description="Optional incident title")
    description: Optional[str] = Field(None, description="Citizen report text")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    address_text: Optional[str] = None


class StatusHistorySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    old_status: Optional[str]
    new_status: str
    changed_by_user_id: Optional[int]
    notes: Optional[str]
    created_at: datetime


class IncidentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: Optional[str]
    disaster_type: str
    priority_score: float
    priority_level: str
    confidence: float
    status: str
    image_url: Optional[str]
    language_detected: str
    created_by_id: Optional[int]
    assigned_responder_id: Optional[int]
    assigned_team_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    location: Optional[LocationSchema]


class IncidentDetailResponse(IncidentResponse):
    class_probabilities: Dict[str, float]
    recommendations: List[str]
    status_history: List[StatusHistorySchema]


class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    assigned_responder_id: Optional[int] = None
    assigned_team_id: Optional[int] = None
    notes: Optional[str] = None
