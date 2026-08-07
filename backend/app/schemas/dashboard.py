from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, ConfigDict


class DashboardStats(BaseModel):
    total_incidents: int
    critical: int
    high: int
    medium: int
    low: int
    very_low: int
    pending: int
    assigned: int
    en_route: int
    in_progress: int
    completed: int


class RecentIncident(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    disaster_type: str
    priority_score: float
    priority_level: str
    status: str
    created_at: datetime
    location_summary: Optional[str] = None


class MapLocation(BaseModel):
    id: int
    title: str
    latitude: float
    longitude: float
    priority_score: float
    priority_level: str
    status: str
    disaster_type: str


class AnalyticsResponse(BaseModel):
    disaster_distribution: Dict[str, int]
    priority_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    daily_incidents: Dict[str, int]
    average_confidence: float
