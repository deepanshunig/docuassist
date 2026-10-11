import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from backend.config import Settings
from backend.errors import DocumentError
from backend.ingestion import DocumentService
from backend.schemas import DeleteResult, Document, DocumentPages, UploadResult
from backend.storage import Storage

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    service = DocumentService(settings, Storage(settings.database_path))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service.initialize()
        yield

    application = FastAPI(
        title="DocuAssist", version="0.2.1",
        description="PDF upload, local OCR, persistent page extraction and inspection.",
        lifespan=lifespan,
    )
    application.state.document_service = service

    @application.exception_handler(DocumentError)
    async def handle_document_error(request: Request, error: DocumentError):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception):
        logger.error("Document operation failed", exc_info=error)
        return JSONResponse(status_code=500, content={"error": {
            "code": "operation_failed", "message": "The operation could not be completed. "
            "Check the backend log and try again."
        }})

    @application.get("/health")
    def health():
        ocr_status = service.ocr.status()
        return {
            "status": "ok", "version": "0.2.1", "milestone": "document_ingestion",
            "document_count": len(service.storage.list_documents()),
            "limits": {"max_upload_mb": settings.max_upload_mb,
                       "max_pages": settings.max_pages,
                       "max_documents": settings.max_documents,
                       "max_ocr_pages": settings.max_ocr_pages},
            "ocr": ocr_status,
            "capabilities": {"pdf_extraction": True,
                             "ocr": ocr_status["state"] in ("ready", "not_loaded"),
                             "semantic_search": False,
                             "generated_answers": False},
        }

    @application.post("/documents", response_model=UploadResult, status_code=201)
    def upload_document(file: Annotated[UploadFile, File()], response: Response):
        try:
            content = file.file.read(settings.max_upload_bytes + 1)
            result = service.upload(file.filename, content)
        finally:
            file.file.close()
        if result.duplicate:
            response.status_code = 200
        return result

    @application.get("/documents", response_model=list[Document])
    def list_documents():
        return service.storage.list_documents()

    @application.get("/documents/{document_id}/pages", response_model=DocumentPages)
    def document_pages(document_id: str):
        return service.pages(document_id)

    @application.post("/documents/{document_id}/reextract", response_model=DocumentPages)
    def reextract_document(document_id: str):
        return service.reextract(document_id)

    @application.delete("/documents/{document_id}", response_model=DeleteResult)
    def delete_document(document_id: str):
        return service.delete(document_id)

    return application


app = create_app()
