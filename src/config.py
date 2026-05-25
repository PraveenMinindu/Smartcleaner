import os
from dotenv import load_dotenv

_env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".env"
)
load_dotenv(dotenv_path=_env_path)


class Settings:
    API_PORT: int    = int(os.getenv("API_PORT", "8000"))
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    KNN_NEIGHBORS: int           = int(os.getenv("KNN_NEIGHBORS", "5"))
    OUTLIER_CONTAMINATION: float = float(os.getenv("OUTLIER_CONTAMINATION", "0.05"))
    SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "data/schema.json")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    def display(self) -> None:
        print(f"\nEnvironment: {self.ENVIRONMENT}")
        print(f"Max File Size: {self.MAX_FILE_SIZE_MB} MB")
        print(f"KNN Neighbors: {self.KNN_NEIGHBORS}")


settings = Settings()