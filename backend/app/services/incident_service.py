from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.repositories.incident_repository import IncidentRepository
from app.repositories.audit_repository import AuditRepository
from app.services.ml_service import PredictionService, get_prediction_service
from app.utils.recommendations import get_response_recommendations
from app.models.incident import Incident
from app.models.prediction import Prediction
from app.models.location import Location
from app.models.status_history import StatusHistory


class IncidentService:
    def __init__(self, db: Session, prediction_service: Optional[PredictionService] = None):
        self.db = db
        self.incident_repo = IncidentRepository(db)
        self.audit_repo = AuditRepository(db)
        self.prediction_service = prediction_service or get_prediction_service()

    def create_incident(
        self,
        description: Optional[str],
        latitude: float,
        longitude: float,
        address_text: Optional[str] = None,
        title: Optional[str] = None,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        user_id: Optional[int] = None,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Run prediction through PredictionService
        pred_res = self.prediction_service.run_prediction(
            image_input=image_path,
            text_report=description,
        )

        # Generate title if not supplied
        disaster_type = pred_res["disaster_type"]
        priority_level = pred_res["priority_level"]
        generated_title = title if title and title.strip() else f"{priority_level} Priority {disaster_type} Incident"

        if not application_context:
            application_context = "Operations Center" if user_id else "Citizen Portal"

        # Create Incident record & related entities
        incident = self.incident_repo.create(
            title=generated_title,
            description=description,
            disaster_type=disaster_type,
            priority_score=pred_res["priority_score"],
            priority_level=priority_level,
            confidence=pred_res["confidence"],
            language_detected=pred_res["language_detected"],
            image_url=image_url,
            created_by_id=user_id,
            latitude=latitude,
            longitude=longitude,
            address_text=address_text,
            class_probabilities=pred_res["class_probabilities"],
            application_context=application_context,
        )

        self.audit_repo.log_action(
            action="INCIDENT_CREATE",
            endpoint="/incidents",
            user_id=user_id,
            details={
                "incident_id": incident.id,
                "disaster_type": disaster_type,
                "priority_score": pred_res["priority_score"],
            },
        )

        return self.get_incident_detail(incident.id)

    def get_incident_detail(self, incident_id: int) -> Optional[Dict[str, Any]]:
        incident = self.incident_repo.get_by_id(incident_id)
        if not incident:
            return None

        # Fetch latest prediction for class probabilities
        pred = self.db.query(Prediction).filter(Prediction.incident_id == incident.id).order_by(Prediction.created_at.desc()).first()
        class_probabilities = pred.class_probabilities if pred else {}

        # Get response recommendations
        recommendations = get_response_recommendations(incident.disaster_type, incident.priority_level)

        # Build response dict
        return {
            "id": incident.id,
            "title": incident.title,
            "description": incident.description,
            "disaster_type": incident.disaster_type,
            "priority_score": incident.priority_score,
            "priority_level": incident.priority_level,
            "confidence": incident.confidence,
            "status": incident.status,
            "image_url": incident.image_url,
            "language_detected": incident.language_detected,
            "application_context": incident.application_context,
            "created_by_id": incident.created_by_id,
            "assigned_responder_id": incident.assigned_responder_id,
            "assigned_team_id": incident.assigned_team_id,
            "created_at": incident.created_at,
            "updated_at": incident.updated_at,
            "location": {
                "latitude": incident.location.latitude if incident.location else 0.0,
                "longitude": incident.location.longitude if incident.location else 0.0,
                "address_text": incident.location.address_text if incident.location else None,
            } if incident.location else None,
            "class_probabilities": class_probabilities,
            "recommendations": recommendations,
            "status_history": incident.status_history,
        }

    def update_incident_status(
        self,
        incident_id: int,
        new_status: str,
        user_id: Optional[int] = None,
        notes: Optional[str] = None,
        assigned_responder_id: Optional[int] = None,
        assigned_team_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        updated = self.incident_repo.update_status(
            incident_id=incident_id,
            new_status=new_status,
            user_id=user_id,
            notes=notes,
            assigned_responder_id=assigned_responder_id,
            assigned_team_id=assigned_team_id,
        )
        if not updated:
            return None

        self.audit_repo.log_action(
            action="INCIDENT_STATUS_UPDATE",
            endpoint=f"/incidents/{incident_id}",
            user_id=user_id,
            details={"incident_id": incident_id, "new_status": new_status, "notes": notes},
        )
        return self.get_incident_detail(incident_id)
