from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUASSIST_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    data_dir: Path = PROJECT_ROOT / "data"
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_pages: int = Field(default=50, ge=1, le=500)
    max_documents: int = Field(default=10, ge=1, le=1000)
    ocr_enabled: bool = True
    ocr_dpi: int = Field(default=180, ge=72, le=300)
    max_ocr_pages: int = Field(default=10, ge=1, le=500)
    ocr_max_pixels: int = Field(default=4_000_000, ge=100_000, le=16_000_000)

    @field_validator("data_dir")
    @classmethod
    def resolve_data_directory(cls, value: Path) -> Path:
        return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def database_path(self) -> Path:
        return self.data_dir / "docuassist.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"
