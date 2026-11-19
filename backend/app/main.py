import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root and backend directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine, Base, get_db
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.error_handler import global_exception_handler
from app.services.auth_service import seed_initial_users_and_teams
from app.services.ml_service import initialize_ml_models, get_prediction_service
from app.routers import (
    auth_router,
    predict_router,
    incidents_router,
    responder_router,
    admin_router,
    dashboard_router,
)

SERVER_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Create database tables
    Base.metadata.create_all(bind=engine)
    
    # 2. Run database migrations for new columns
    db = Session(bind=engine)
    try:
        # Check if application_context exists in incidents table
        columns = [row[1] for row in db.execute(text("PRAGMA table_info(incidents)")).fetchall()]
        if "application_context" not in columns:
            db.execute(text("ALTER TABLE incidents ADD COLUMN application_context VARCHAR DEFAULT 'Citizen Portal'"))
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    # 3. Seed default admin, responder & rescue team
    db = Session(bind=engine)
    try:
        seed_initial_users_and_teams(db)
    finally:
        db.close()

    # 3. Load ML models ONCE during startup
    initialize_ml_models()
    
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Research-prototype disaster triage API. Predictions require loaded, trained ML models.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Exception handlers & middleware
app.add_exception_handler(Exception, global_exception_handler)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded disaster images statically
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include API routers
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(incidents_router)
app.include_router(responder_router)
app.include_router(admin_router)
app.include_router(dashboard_router)


@app.get("/", tags=["API Info"])
def root_info():
    """
    Return API metadata, version, and endpoints overview.
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "incidents": "/incidents",
            "responder": "/responder/incidents",
            "admin": "/admin/statistics",
            "dashboard": "/dashboard/stats",
        },
    }


@app.get("/health", tags=["API Info"])
def health_check(db: Session = Depends(get_db)):
    """
    System health check.
    Returns status, database connection, models_loaded, and server uptime.
    """
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    prediction_service = get_prediction_service()
    adapter = prediction_service.predictor
    vision_loaded = getattr(adapter, "_predictor", None) is not None
    text_loaded = vision_loaded or getattr(adapter, "_text_predictor", None) is not None
    modes_available = (["text", "image", "multimodal"] if vision_loaded
                       else ["text"] if text_loaded else [])

    uptime = round(time.time() - SERVER_START_TIME, 2)
    return {
        "status": "ok" if db_status == "connected" and vision_loaded
                  else "degraded",
        "database": db_status,
        # Legacy flag retains its meaning: full multimodal ensemble is loaded.
        "models_loaded": vision_loaded,
        "text_model_loaded": text_loaded,
        "vision_model_loaded": vision_loaded,
        "available_inference_modes": modes_available,
        "model_variant": "free-tier-supervised-retrieval" if settings.LITE_MODEL_ENABLED else "full-research-model",
        "storage_persistence": "ephemeral" if settings.LITE_MODEL_ENABLED else "configured",
        "uptime_seconds": uptime,
    }
