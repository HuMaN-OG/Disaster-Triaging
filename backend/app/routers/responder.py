from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, require_roles
from app.schemas.incident import IncidentResponse, IncidentDetailResponse
from app.schemas.responder import StatusUpdateRequest, AcceptIncidentRequest
from app.services.incident_service import IncidentService
from app.repositories.incident_repository import IncidentRepository
from app.models.user import User

router = APIRouter(prefix="/responder", tags=["Responder Workflow"])


@router.get("/incidents", response_model=List[IncidentResponse])
def get_pending_responder_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Responder", "Admin"])),
):
    """
    Get pending & assigned incidents for Responder dashboard.
    """
    repo = IncidentRepository(db)
    # Fetch pending or incidents assigned to current responder
    pending_incidents = repo.get_all(skip=skip, limit=limit, status="Pending")
    return pending_incidents


@router.post("/{id}/accept", response_model=IncidentDetailResponse)
def accept_incident(
    id: int,
    request: Optional[AcceptIncidentRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Responder", "Admin"])),
):
    """
    Responder accepts a pending disaster incident.
    Transitions status: Pending -> Assigned.
    """
    incident_service = IncidentService(db)
    incident_dict = incident_service.get_incident_detail(id)
    if not incident_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {id} not found",
        )

    notes = request.notes if request and request.notes else "Accepted by responder"
    team_id = request.team_id if request else None

    updated = incident_service.update_incident_status(
        incident_id=id,
        new_status="Assigned",
        user_id=current_user.id,
        notes=notes,
        assigned_responder_id=current_user.id,
        assigned_team_id=team_id,
    )
    return updated


@router.patch("/{id}/status", response_model=IncidentDetailResponse)
def update_responder_status(
    id: int,
    update_in: StatusUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Responder", "Admin"])),
):
    """
    Update incident rescue status throughout workflow:
    Assigned -> En Route -> Rescue In Progress -> Completed.
    """
    allowed_statuses = {"Assigned", "En Route", "Rescue In Progress", "Completed"}
    if update_in.status not in allowed_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{update_in.status}'. Allowed: {', '.join(allowed_statuses)}",
        )

    incident_service = IncidentService(db)
    updated = incident_service.update_incident_status(
        incident_id=id,
        new_status=update_in.status,
        user_id=current_user.id,
        notes=update_in.notes,
        assigned_responder_id=current_user.id,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {id} not found",
        )
    return updated
