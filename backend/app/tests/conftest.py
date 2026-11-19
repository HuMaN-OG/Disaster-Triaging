import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.core.database import Base, get_db
from app.services.auth_service import seed_initial_users_and_teams
from app.services.ml_service import initialize_ml_models
from app.core.config import settings

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_disaster.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_database_and_models():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Explicitly enable demo fixtures only for tests; production defaults are off.
    settings.DEMO_ACCOUNTS_ENABLED = True
    settings.DEMO_ADMIN_PASSWORD = "Admin@123"
    settings.DEMO_RESPONDER_PASSWORD = "Responder@123"
    db = TestingSessionLocal()
    seed_initial_users_and_teams(db)
    db.close()

    # Preload models
    initialize_ml_models()
    
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
