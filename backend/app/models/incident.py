from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    disaster_type = Column(String, nullable=False)  # Flood, Earthquake, Urban Fire, Wild Fire, Landslide, Drought, Unknown
    priority_score = Column(Float, nullable=False, default=50.0)
    priority_level = Column(String, nullable=False, default="Medium")  # Very Low, Low, Medium, High, Critical
    confidence = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False, default="Pending")  # Pending, Assigned, En Route, Rescue In Progress, Completed
    image_url = Column(String, nullable=True)
    language_detected = Column(String, nullable=False, default="English")
    application_context = Column(String, nullable=True, default="Citizen Portal")
    
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_responder_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_id], back_populates="incidents_created")
    assigned_responder = relationship("User", foreign_keys=[assigned_responder_id], back_populates="incidents_assigned")
    assigned_team = relationship("Team", back_populates="incidents")
    
    predictions = relationship("Prediction", back_populates="incident", cascade="all, delete-orphan")
    location = relationship("Location", back_populates="incident", uselist=False, cascade="all, delete-orphan")
    assignments = relationship("Assignment", back_populates="incident", cascade="all, delete-orphan")
    status_history = relationship("StatusHistory", back_populates="incident", cascade="all, delete-orphan")
