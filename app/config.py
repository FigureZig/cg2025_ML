import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Finance API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/data/finance.db"

    ML_ARTIFACTS_DIR: Path = BASE_DIR / "app" / "ml" / "artifacts"
    MODEL_PATH: Path = ML_ARTIFACTS_DIR / "catboost_classifier.cbm"
    ANOMALY_MODEL_PATH: Path = ML_ARTIFACTS_DIR / "anomaly_detector.pkl"
    SCALER_PATH: Path = ML_ARTIFACTS_DIR / "anomaly_scaler.pkl"

    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    MOCK_NOW: datetime = datetime(2023, 12, 13, 23, 59, 59)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()