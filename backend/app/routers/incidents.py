from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user_optional, get_current_user, require_roles
from app.schemas.incident import (
    IncidentResponse,
    IncidentDetailResponse,
    IncidentUpdate,
    ResponseBriefResponse,
    TimelineEvent,
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
    application_context: Optional[str] = Form(None),
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
        application_context=application_context,
    )
    return result


@router.get("", response_model=List[IncidentResponse])
def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    disaster_type: Optional[str] = Query(None),
    priority_level: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    application_context: Optional[str] = Query(None),
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
        location=location,
        application_context=application_context,
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


@router.get("/{id}/brief", response_model=ResponseBriefResponse)
def get_incident_brief(id: int, db: Session = Depends(get_db)):
    """
    Generate Incident Response Brief with Confirmed Data and AI/Derived Recommendations.
    """
    incident_service = IncidentService(db)
    detail = incident_service.get_incident_detail(id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {id} not found",
        )

    disaster_type = detail["disaster_type"]
    priority_level = detail["priority_level"]

    # Deterministic recommendations
    # Risk summary
    risk_summaries = {
        "Flood": "High risk of rising water levels, structural damage, waterborne disease outbreaks, and isolation of residents. Priority target for evacuation.",
        "Urban Fire": "Extreme threat to life safety, risk of structural collapse, flashovers, and smoke inhalation. Rapid spread probability is high.",
        "Wild Fire": "Fast-spreading fire threat to residences, smoke inhalation hazard, and emergency responder access blocks.",
        "Earthquake": "High danger of structural collapses, gas leaks, road blockages, and aftershocks. Survivors might be trapped under rubble.",
        "Landslide": "Mudflow and rocks blocking roads. Risk of further slope instability and collapse of nearby buildings.",
        "Drought": "Long-term threat to agriculture, severe water scarcity, increased wildfire risk, and supply chain disruptions."
    }
    risk_summary = risk_summaries.get(disaster_type, "General emergency situation requiring local responder assessment. Threat levels to be monitored.")

    # Immediate actions
    immediate_actions_map = {
        "Flood": ["Establish water rescue teams", "Alert local residents to move to upper floors", "Activate flood barrier checkpoints"],
        "Urban Fire": ["Dispatch local fire rescue units", "Establish evacuation perimeter", "Ensure gas supply lines are cut"],
        "Wild Fire": ["Establish firebreaks", "Evacuate downwind areas", "Deploy aerial retardants"],
        "Earthquake": ["Initiate K9 search and rescue operations", "Shut down main power grids", "Deploy heavy clearing machinery"],
        "Landslide": ["Clear debris with bulldozer support", "Evacuate downhill residences", "Monitor geological sensors"],
        "Drought": ["Ration local water distribution", "Distribute crop emergency packages", "Pre-position fire retardant units"]
    }
    immediate_actions = immediate_actions_map.get(disaster_type, ["Dispatch generic assessment team", "Verify citizen report details", "Establish communication line"])

    # Recommended resources
    recommended_resources_map = {
        "Flood": ["Inflatable boats", "Life jackets", "Water pumps", "Emergency medical kits"],
        "Urban Fire": ["Fire engines", "Oxygen masks", "Thermal cameras", "First aid equipment"],
        "Wild Fire": ["Fire retardants", "Water tenders", "Bulldozers", "Helicopters"],
        "Earthquake": ["Concrete cutters", "Search dogs", "Portable power generators", "Medivac units"],
        "Landslide": ["Excavators", "Soil stabilization gear", "Geotechnical sensors", "Helicopters"],
        "Drought": ["Water tankers", "Famine relief packages", "Fire management tools"]
    }
    recommended_resources = recommended_resources_map.get(disaster_type, ["Basic patrol vehicle", "Radio kit", "First-aid pack"])

    # Build timeline events from status history
    timeline = []
    # Sort history by created_at ascending
    for hist in sorted(detail["status_history"], key=lambda h: h.created_at):
        timeline.append(TimelineEvent(
            timestamp=hist.created_at,
            status=hist.new_status,
            notes=hist.notes
        ))

    # Fetch responder and team names if assigned
    assigned_responder = None
    if detail["assigned_responder_id"]:
        from app.models.user import User
        user = db.query(User).filter(User.id == detail["assigned_responder_id"]).first()
        if user:
            assigned_responder = user.full_name

    assigned_team = None
    if detail["assigned_team_id"]:
        from app.models.team import Team
        team = db.query(Team).filter(Team.id == detail["assigned_team_id"]).first()
        if team:
            assigned_team = team.name

    return ResponseBriefResponse(
        incident_id=detail["id"],
        disaster_type=detail["disaster_type"],
        priority_score=detail["priority_score"],
        priority_level=detail["priority_level"],
        confidence=detail["confidence"],
        citizen_report=detail["description"],
        detected_language=detail["language_detected"],
        location=detail["location"]["address_text"] if detail["location"] else None,
        latitude=detail["location"]["latitude"] if detail["location"] else 0.0,
        longitude=detail["location"]["longitude"] if detail["location"] else 0.0,
        created_at=detail["created_at"],
        current_status=detail["status"],
        assigned_responder=assigned_responder,
        assigned_team=assigned_team,
        timeline=timeline,
        risk_summary=risk_summary,
        immediate_actions=immediate_actions,
        recommended_resources=recommended_resources
    )
