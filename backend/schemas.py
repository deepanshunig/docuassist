from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Page(BaseModel):
    pdf_page: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    warning: str | None = None
    extraction_method: Literal["text", "ocr", "none"] = "none"


class Document(BaseModel):
    id: str
    filename: str
    sha256: str
    size_bytes: int
    page_count: int
    text_page_count: int
    status: Literal["extracted", "extracted_with_warnings"]
    warnings: list[str]
    created_at: str


class UploadResult(BaseModel):
    document: Document
    duplicate: bool = False
    indexed: bool = False
    indexing_warning: str | None = None


class DocumentPages(BaseModel):
    document: Document
    pages: list[Page]


class DeleteResult(BaseModel):
    deleted: bool
    document_id: str
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    filename: str
    pdf_page: int
    ordinal: int
    text: str
    token_count: int
    vector: bytes
    dimension: int
    fingerprint: str
    extraction_method: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a question")
        return value


class Source(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    pdf_page: int
    passage: str
    score: float
    extraction_method: Literal["text", "ocr", "none"]


class SearchResult(BaseModel):
    sources: list[Source]
    collection_version: int
    embedding_model: str
    latency_ms: float


class GeneratedAnswer(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["answered", "insufficient_evidence", "ambiguous", "clarification"]
    answer: str = Field(min_length=1, max_length=6000)
    citations: list[str] = Field(max_length=10)


class AnswerResult(BaseModel):
    status: Literal["answered", "insufficient_evidence", "ambiguous", "clarification"]
    answer: str
    sources: list[Source]
    retrieved_sources: list[Source]
    request_id: str
    collection_version: int
    model: str | None
    latency_ms: float
    timings_ms: dict[str, float]
