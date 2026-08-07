from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, EmailStr, ConfigDict


class TeamCreate(BaseModel):
    name: str
    specialization: str
    contact_number: Optional[str] = None


class TeamResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    specialization: str
    contact_number: Optional[str]
    is_active: bool
    created_at: datetime


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int]
    action: str
    endpoint: str
    details: Optional[Any]
    ip_address: Optional[str]
    created_at: datetime


class SystemStatsResponse(BaseModel):
    total_users: int
    total_responders: int
    total_citizens: int
    total_teams: int
    total_incidents: int
    active_incidents: int
    completed_incidents: int
