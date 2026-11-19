from pathlib import Path
from secrets import token_urlsafe
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Disaster Triaging System API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = ""
    
    # JWT Auth settings
    # Provide a random, process-local signing key when none is configured.
    # Production must set JWT_SECRET so tokens survive restarts.
    SECRET_KEY: str = Field(default_factory=lambda: token_urlsafe(48), validation_alias="JWT_SECRET")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Demo accounts are disabled by default and must be explicitly configured
    # for local demonstrations only. Never enable with public credentials.
    DEMO_ACCOUNTS_ENABLED: bool = False
    # Demo User Accounts Configuration
    DEMO_RESPONDER_EMAIL: str = "responder@disaster.org"
    DEMO_RESPONDER_PASSWORD: str = ""
    DEMO_ADMIN_EMAIL: str = "admin@disaster.org"
    DEMO_ADMIN_PASSWORD: str = ""
    
    # Database settings
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/disaster_triage.db"
    
    # Upload settings
    UPLOAD_DIR: Path = BASE_DIR / "backend" / "app" / "uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:80",
    ]
    
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
