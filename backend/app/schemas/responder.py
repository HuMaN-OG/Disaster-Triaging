from typing import Optional
from pydantic import BaseModel
from app.schemas.incident import IncidentResponse


class StatusUpdateRequest(BaseModel):
    status: str  # Assigned, En Route, Rescue In Progress, Completed
    notes: Optional[str] = None


class AcceptIncidentRequest(BaseModel):
    team_id: Optional[int] = None
    notes: Optional[str] = None


class ResponderIncidentResponse(IncidentResponse):
    pass
