from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, default="Citizen", nullable=False)  # Citizen, Responder, Admin
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    incidents_created = relationship("Incident", foreign_keys="Incident.created_by_id", back_populates="created_by")
    incidents_assigned = relationship("Incident", foreign_keys="Incident.assigned_responder_id", back_populates="assigned_responder")
    assignments = relationship("Assignment", back_populates="responder")
    notifications = relationship("Notification", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
