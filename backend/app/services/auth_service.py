from typing import Optional
from sqlalchemy.orm import Session
from app.repositories.user_repository import UserRepository
from app.repositories.audit_repository import AuditRepository
from app.core.security import verify_password, create_access_token
from app.models.user import User


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
        self.audit_repo = AuditRepository(db)

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        user = self.user_repo.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def create_user_token(self, user: User) -> dict:
        token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
        access_token = create_access_token(data=token_data)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "role": user.role,
            "user_id": user.id,
            "full_name": user.full_name,
        }

    def register_user(self, email: str, password: str, full_name: str, role: str = "Citizen") -> User:
        existing = self.user_repo.get_by_email(email)
        if existing:
            raise ValueError(f"User with email '{email}' already exists")
        
        user = self.user_repo.create(email=email, password=password, full_name=full_name, role=role)
        self.audit_repo.log_action(
            action="USER_REGISTER",
            endpoint="/auth/register",
            user_id=user.id,
            details={"email": email, "role": role},
        )
        return user


def seed_initial_users_and_teams(db: Session) -> None:
    """Ensure deterministic Admin, Responder, and Team exist on startup."""
    from app.core.config import settings
    user_repo = UserRepository(db)
    
    # Default Admin
    admin_user = user_repo.get_by_email(settings.DEMO_ADMIN_EMAIL)
    if not admin_user:
        user_repo.create(
            email=settings.DEMO_ADMIN_EMAIL,
            password=settings.DEMO_ADMIN_PASSWORD,
            full_name="System Administrator",
            role="Admin",
        )
    elif not verify_password(settings.DEMO_ADMIN_PASSWORD, admin_user.hashed_password):
        user_repo.update(admin_user.id, password=settings.DEMO_ADMIN_PASSWORD)

    # Default Responder
    responder_user = user_repo.get_by_email(settings.DEMO_RESPONDER_EMAIL)
    if not responder_user:
        user_repo.create(
            email=settings.DEMO_RESPONDER_EMAIL,
            password=settings.DEMO_RESPONDER_PASSWORD,
            full_name="Lead Disaster Responder",
            role="Responder",
        )
    elif not verify_password(settings.DEMO_RESPONDER_PASSWORD, responder_user.hashed_password):
        user_repo.update(responder_user.id, password=settings.DEMO_RESPONDER_PASSWORD)

    # Default Rescue Team
    from app.models.team import Team
    if not db.query(Team).filter(Team.name == "Alpha Search & Rescue").first():
        team = Team(
            name="Alpha Search & Rescue",
            specialization="Rapid Search and Extraction",
            contact_number="+1-800-RESCUE-1",
            is_active=True,
        )
        db.add(team)
        db.commit()
