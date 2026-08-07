from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_roles
from app.schemas.auth import UserResponse, UserCreate, UserUpdate
from app.schemas.admin import TeamCreate, TeamResponse, AuditLogResponse, SystemStatsResponse
from app.schemas.incident import IncidentResponse
from app.repositories.user_repository import UserRepository
from app.repositories.incident_repository import IncidentRepository
from app.repositories.audit_repository import AuditRepository
from app.models.user import User
from app.models.team import Team

router = APIRouter(prefix="/admin", tags=["Admin Operations"])


# User Management
@router.get("/users", response_model=List[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1),
    role: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    repo = UserRepository(db)
    return repo.get_all(skip=skip, limit=limit, role=role)


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    repo = UserRepository(db)
    if repo.get_by_email(user_in.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return repo.create(
        email=user_in.email,
        password=user_in.password,
        full_name=user_in.full_name,
        role=user_in.role or "Citizen",
    )


@router.patch("/users/{id}", response_model=UserResponse)
def update_user(
    id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    repo = UserRepository(db)
    updated = repo.update(id, **user_update.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return updated


@router.delete("/users/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    repo = UserRepository(db)
    if not repo.delete(id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return None


# Team Management
@router.post("/teams", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
def create_team(
    team_in: TeamCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    existing = db.query(Team).filter(Team.name == team_in.name).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Team name already exists")
    team = Team(
        name=team_in.name,
        specialization=team_in.specialization,
        contact_number=team_in.contact_number,
    )
    db.add(team)
    db.commit()
    db.refresh(team)
    return team


@router.get("/teams", response_model=List[TeamResponse])
def list_teams(
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin", "Responder"])),
):
    return db.query(Team).filter(Team.is_active == True).all()


# Audit Logs & Statistics
@router.get("/audit-logs", response_model=List[AuditLogResponse])
def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1),
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    repo = AuditRepository(db)
    return repo.get_logs(skip=skip, limit=limit)


@router.get("/statistics", response_model=SystemStatsResponse)
def get_system_statistics(
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles(["Admin"])),
):
    total_users = db.query(User).count()
    total_responders = db.query(User).filter(User.role == "Responder").count()
    total_citizens = db.query(User).filter(User.role == "Citizen").count()
    total_teams = db.query(Team).count()
    
    inc_repo = IncidentRepository(db)
    total_incidents = inc_repo.count_total()
    active_incidents = len(inc_repo.get_all(status="Assigned")) + len(inc_repo.get_all(status="En Route")) + len(inc_repo.get_all(status="Rescue In Progress"))
    completed_incidents = len(inc_repo.get_all(status="Completed"))

    return SystemStatsResponse(
        total_users=total_users,
        total_responders=total_responders,
        total_citizens=total_citizens,
        total_teams=total_teams,
        total_incidents=total_incidents,
        active_incidents=active_incidents,
        completed_incidents=completed_incidents,
    )
