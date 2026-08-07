from app.routers.auth import router as auth_router
from app.routers.predict import router as predict_router
from app.routers.incidents import router as incidents_router
from app.routers.responder import router as responder_router
from app.routers.admin import router as admin_router
from app.routers.dashboard import router as dashboard_router

__all__ = [
    "auth_router",
    "predict_router",
    "incidents_router",
    "responder_router",
    "admin_router",
    "dashboard_router",
]
