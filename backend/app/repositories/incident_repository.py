from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.incident import Incident
from app.models.prediction import Prediction
from app.models.location import Location
from app.models.status_history import StatusHistory
from app.models.assignment import Assignment


class IncidentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        title: str,
        description: Optional[str],
        disaster_type: str,
        priority_score: float,
        priority_level: str,
        confidence: float,
        language_detected: str,
        image_url: Optional[str],
        created_by_id: Optional[int],
        latitude: float,
        longitude: float,
        address_text: Optional[str],
        class_probabilities: Dict[str, float],
    ) -> Incident:
        # Create Incident entity
        incident = Incident(
            title=title,
            description=description,
            disaster_type=disaster_type,
            priority_score=priority_score,
            priority_level=priority_level,
            confidence=confidence,
            status="Pending",
            image_url=image_url,
            language_detected=language_detected,
            created_by_id=created_by_id,
        )
        self.db.add(incident)
        self.db.flush()  # Get generated incident.id

        # Create Location entity
        location = Location(
            incident_id=incident.id,
            latitude=latitude,
            longitude=longitude,
            address_text=address_text,
        )
        self.db.add(location)

        # Create Prediction record
        prediction = Prediction(
            incident_id=incident.id,
            disaster_type=disaster_type,
            priority_score=priority_score,
            priority_level=priority_level,
            confidence=confidence,
            class_probabilities=class_probabilities,
            language_detected=language_detected,
        )
        self.db.add(prediction)

        # Initial StatusHistory record
        history = StatusHistory(
            incident_id=incident.id,
            old_status=None,
            new_status="Pending",
            changed_by_user_id=created_by_id,
            notes="Incident reported and registered in system",
        )
        self.db.add(history)

        self.db.commit()
        self.db.refresh(incident)
        return incident

    def get_by_id(self, incident_id: int) -> Optional[Incident]:
        return self.db.query(Incident).filter(Incident.id == incident_id).first()

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        disaster_type: Optional[str] = None,
        priority_level: Optional[str] = None,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> List[Incident]:
        query = self.db.query(Incident)

        if disaster_type:
            query = query.filter(Incident.disaster_type == disaster_type)
        if priority_level:
            query = query.filter(Incident.priority_level == priority_level)
        if status:
            query = query.filter(Incident.status == status)

        # Sorting
        sort_attr = getattr(Incident, sort_by, Incident.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_attr))
        else:
            query = query.order_by(sort_attr)

        return query.offset(skip).limit(limit).all()

    def count_total(self) -> int:
        return self.db.query(Incident).count()

    def update_status(
        self,
        incident_id: int,
        new_status: str,
        user_id: Optional[int] = None,
        notes: Optional[str] = None,
        assigned_responder_id: Optional[int] = None,
        assigned_team_id: Optional[int] = None,
    ) -> Optional[Incident]:
        incident = self.get_by_id(incident_id)
        if not incident:
            return None

        old_status = incident.status
        incident.status = new_status

        if assigned_responder_id is not None:
            incident.assigned_responder_id = assigned_responder_id
        if assigned_team_id is not None:
            incident.assigned_team_id = assigned_team_id

        # Status History entry
        history = StatusHistory(
            incident_id=incident.id,
            old_status=old_status,
            new_status=new_status,
            changed_by_user_id=user_id,
            notes=notes,
        )
        self.db.add(history)

        # Assignment entry if responder assigned
        if assigned_responder_id:
            assignment = Assignment(
                incident_id=incident.id,
                responder_id=assigned_responder_id,
                team_id=assigned_team_id,
                notes=notes,
            )
            self.db.add(assignment)

        self.db.commit()
        self.db.refresh(incident)
        return incident

    def delete(self, incident_id: int) -> bool:
        incident = self.get_by_id(incident_id)
        if not incident:
            return False
        self.db.delete(incident)
        self.db.commit()
        return True
