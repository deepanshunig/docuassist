from pathlib import Path

from typing import Literal

from pydantic import Field, field_validator, model_validator
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
    rag_enabled: bool = True
    embedding_dir: Path | None = None
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_threads: int = Field(default=4, ge=1, le=32)
    chunk_tokens: int = Field(default=200, ge=16, le=254)
    chunk_overlap: int = Field(default=30, ge=0, le=100)
    retrieval_k: int = Field(default=5, ge=1, le=10)
    retrieval_min_score: float = Field(default=0.20, ge=0, le=1)
    llm_provider: Literal["llamacpp", "ollama"] = "llamacpp"
    llm_url: str = "http://127.0.0.1:8012"
    llm_model: str = "qwen2.5-1.5b-instruct"
    llm_context_tokens: int = Field(default=4096, ge=2048, le=32768)
    llm_output_tokens: int = Field(default=512, ge=128, le=2048)
    llm_timeout_seconds: int = Field(default=180, ge=5, le=600)

    @model_validator(mode="after")
    def check_rag_limits(self):
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        if self.llm_output_tokens + 1024 >= self.llm_context_tokens:
            raise ValueError("Leave at least 1024 context tokens for the question and evidence")
        return self

    @field_validator("data_dir", "embedding_dir")
    @classmethod
    def resolve_data_directory(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
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

    @property
    def model_dir(self) -> Path:
        return (self.embedding_dir or self.data_dir / "models" / "minilm").resolve()
