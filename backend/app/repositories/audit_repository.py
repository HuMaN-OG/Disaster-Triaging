from typing import Optional, List, Any
from sqlalchemy.orm import Session
from app.models.audit_log import AuditLog


class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_action(
        self,
        action: str,
        endpoint: str,
        user_id: Optional[int] = None,
        details: Optional[Any] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        audit = AuditLog(
            user_id=user_id,
            action=action,
            endpoint=endpoint,
            details=details,
            ip_address=ip_address,
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(audit)
        return audit

    def get_logs(self, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        return self.db.query(AuditLog).order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()
