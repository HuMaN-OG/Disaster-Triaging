from typing import List, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import APIRouter, Depends

from app.core.database import get_db
from app.schemas.dashboard import DashboardStats, RecentIncident, MapLocation, AnalyticsResponse
from app.models.incident import Incident
from app.models.location import Location

router = APIRouter(prefix="/dashboard", tags=["Dashboard Analytics"])


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get aggregated counts by priority level and status for real-time dashboard cards.
    """
    total = db.query(Incident).count()
    
    # Priority counts
    critical = db.query(Incident).filter(Incident.priority_level == "Critical").count()
    high = db.query(Incident).filter(Incident.priority_level == "High").count()
    medium = db.query(Incident).filter(Incident.priority_level == "Medium").count()
    low = db.query(Incident).filter(Incident.priority_level == "Low").count()
    very_low = db.query(Incident).filter(Incident.priority_level == "Very Low").count()

    # Status counts
    pending = db.query(Incident).filter(Incident.status == "Pending").count()
    assigned = db.query(Incident).filter(Incident.status == "Assigned").count()
    en_route = db.query(Incident).filter(Incident.status == "En Route").count()
    in_progress = db.query(Incident).filter(Incident.status == "Rescue In Progress").count()
    completed = db.query(Incident).filter(Incident.status == "Completed").count()

    return DashboardStats(
        total_incidents=total,
        critical=critical,
        high=high,
        medium=medium,
        low=low,
        very_low=very_low,
        pending=pending,
        assigned=assigned,
        en_route=en_route,
        in_progress=in_progress,
        completed=completed,
    )


@router.get("/recent", response_model=List[RecentIncident])
def get_recent_incidents(db: Session = Depends(get_db)):
    """
    Get the 10 most recent disaster incidents.
    """
    recent = db.query(Incident).order_by(Incident.created_at.desc()).limit(10).all()
    results = []
    for inc in recent:
        loc_str = inc.location.address_text if inc.location and inc.location.address_text else None
        if not loc_str and inc.location:
            loc_str = f"({inc.location.latitude:.4f}, {inc.location.longitude:.4f})"
        results.append(
            RecentIncident(
                id=inc.id,
                title=inc.title,
                disaster_type=inc.disaster_type,
                priority_score=inc.priority_score,
                priority_level=inc.priority_level,
                status=inc.status,
                created_at=inc.created_at,
                location_summary=loc_str,
            )
        )
    return results


@router.get("/map", response_model=List[MapLocation])
def get_map_locations(db: Session = Depends(get_db)):
    """
    Get geographic coordinates and triage metadata for interactive map rendering.
    """
    incidents = db.query(Incident).join(Location).all()
    map_data = []
    for inc in incidents:
        if inc.location:
            map_data.append(
                MapLocation(
                    id=inc.id,
                    title=inc.title,
                    latitude=inc.location.latitude,
                    longitude=inc.location.longitude,
                    priority_score=inc.priority_score,
                    priority_level=inc.priority_level,
                    status=inc.status,
                    disaster_type=inc.disaster_type,
                )
            )
    return map_data


@router.get("/analytics", response_model=AnalyticsResponse)
def get_dashboard_analytics(db: Session = Depends(get_db)):
    """
    Get breakdown analytics for charts: disaster type distribution, priority levels, daily incident trend, languages, average confidence.
    """
    incidents = db.query(Incident).all()

    disaster_dist: Dict[str, int] = {}
    priority_dist: Dict[str, int] = {}
    status_dist: Dict[str, int] = {}
    lang_dist: Dict[str, int] = {}
    daily_incidents: Dict[str, int] = {}
    total_conf = 0.0

    for inc in incidents:
        # Disaster distribution
        disaster_dist[inc.disaster_type] = disaster_dist.get(inc.disaster_type, 0) + 1
        
        # Priority distribution
        priority_dist[inc.priority_level] = priority_dist.get(inc.priority_level, 0) + 1
        
        # Status distribution
        status_dist[inc.status] = status_dist.get(inc.status, 0) + 1
        
        # Language distribution
        lang_dist[inc.language_detected] = lang_dist.get(inc.language_detected, 0) + 1
        
        # Daily breakdown (YYYY-MM-DD)
        day_str = inc.created_at.strftime("%Y-%m-%d")
        daily_incidents[day_str] = daily_incidents.get(day_str, 0) + 1
        
        total_conf += inc.confidence

    avg_conf = round(total_conf / max(len(incidents), 1), 4)

    return AnalyticsResponse(
        disaster_distribution=disaster_dist,
        priority_distribution=priority_dist,
        status_distribution=status_dist,
        language_distribution=lang_dist,
        daily_incidents=daily_incidents,
        average_confidence=avg_conf,
    )
