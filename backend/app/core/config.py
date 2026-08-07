from pathlib import Path
from pydantic_settings import BaseSettings

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Disaster Triaging System API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = ""
    
    # JWT Auth settings
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database settings
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/disaster_triage.db"
    
    # Upload settings
    UPLOAD_DIR: Path = BASE_DIR / "backend" / "app" / "uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        case_sensitive = True


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
