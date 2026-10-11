from typing import Literal

from pydantic import BaseModel, Field


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


class DocumentPages(BaseModel):
    document: Document
    pages: list[Page]


class DeleteResult(BaseModel):
    deleted: bool
    document_id: str
    warnings: list[str] = Field(default_factory=list)
