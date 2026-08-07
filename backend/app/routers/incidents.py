from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user_optional, get_current_user, require_roles
from app.schemas.incident import (
    IncidentResponse,
    IncidentDetailResponse,
    IncidentUpdate,
)
from app.services.incident_service import IncidentService
from app.utils.file_upload import save_uploaded_image
from app.repositories.incident_repository import IncidentRepository
from app.models.user import User

router = APIRouter(prefix="/incidents", tags=["Incidents"])


@router.post("", response_model=IncidentDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_incident(
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address_text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Create a new disaster incident.
    Accepts Text, Image, and Browser GPS coordinates.
    Automatically executes model inference, stores entities in DB, and sets status to Pending.
    """
    # Validate GPS coordinates
    if not (-90.0 <= latitude <= 90.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid latitude value {latitude}. Must be between -90.0 and 90.0.",
        )
    if not (-180.0 <= longitude <= 180.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid longitude value {longitude}. Must be between -180.0 and 180.0.",
        )

    image_url = None
    saved_temp_path = None
    if image:
        image_url = await save_uploaded_image(image)
        # Get absolute path for predictor
        from app.core.config import settings
        file_name = image_url.split("/")[-1]
        saved_temp_path = str(settings.UPLOAD_DIR / file_name)

    incident_service = IncidentService(db)
    result = incident_service.create_incident(
        description=description,
        latitude=latitude,
        longitude=longitude,
        address_text=address_text,
        title=title,
        image_path=saved_temp_path,
        image_url=image_url,
        user_id=current_user.id if current_user else None,
    )
    return result


@router.get("", response_model=List[IncidentResponse])
def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    disaster_type: Optional[str] = Query(None),
    priority_level: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    db: Session = Depends(get_db),
):
    """
    Get all incidents with pagination, filtering, and sorting.
    """
    repo = IncidentRepository(db)
    incidents = repo.get_all(
        skip=skip,
        limit=limit,
        disaster_type=disaster_type,
        priority_level=priority_level,
        status=status,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return incidents


@router.get("/{id}", response_model=IncidentDetailResponse)
def get_incident(id: int, db: Session = Depends(get_db)):
    """
    Get complete incident by ID including prediction, location, status history, and action recommendations.
    """
    incident_service = IncidentService(db)
    detail = incident_service.get_incident_detail(id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {id} not found",
        )
    return detail


@router.patch("/{id}", response_model=IncidentDetailResponse)
def update_incident(
    id: int,
    update_in: IncidentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update incident status, assigned team, or notes.
    """
    incident_service = IncidentService(db)
    updated = incident_service.update_incident_status(
        incident_id=id,
        new_status=update_in.status if update_in.status else "Pending",
        user_id=current_user.id,
        notes=update_in.notes,
        assigned_responder_id=update_in.assigned_responder_id,
        assigned_team_id=update_in.assigned_team_id,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {id} not found",
        )
    return updated


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_incident(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    """
    Delete incident (Admin authorized only).
    """
    repo = IncidentRepository(db)
    success = repo.delete(id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {id} not found",
        )
    return None
