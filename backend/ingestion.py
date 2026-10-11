import hashlib
import logging
import re
from datetime import datetime, timezone
from io import BytesIO
from threading import RLock
from uuid import UUID, uuid4

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from backend.config import Settings
from backend.errors import DocumentError
from backend.ocr import LocalOCR, OCRUnavailableError
from backend.schemas import DeleteResult, Document, DocumentPages, Page, UploadResult
from backend.storage import Storage

logger = logging.getLogger(__name__)


def display_filename(filename: str | None) -> str:
    name = (filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(character for character in name if character.isprintable()).strip()
    if not name.lower().endswith(".pdf"):
        raise DocumentError("unsupported_file", "Choose a PDF file.", 415)
    return name[:156] + ".pdf" if len(name) > 160 else name


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def extract_pages(content: bytes, settings: Settings, ocr: LocalOCR) -> list[Page]:
    if not content:
        raise DocumentError("empty_file", "The uploaded file is empty.")
    if not content.startswith(b"%PDF-"):
        raise DocumentError("invalid_pdf", "This file does not contain a valid PDF header.")
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise DocumentError("encrypted_pdf", "Upload a PDF without password protection.")
        page_count = len(reader.pages)
        if page_count == 0:
            raise DocumentError("empty_pdf", "This PDF has no pages.")
        if page_count > settings.max_pages:
            raise DocumentError(
                "too_many_pages", f"PDFs may contain at most {settings.max_pages} pages."
            )
        pages = []
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            warning = None
            try:
                raw_text = pdf_page.extract_text() or ""
                text = normalize_text(raw_text)
                if not text:
                    warning = f"PDF page {page_number} has no embedded text."
            except Exception:
                logger.warning("Text extraction failed on PDF page %s", page_number)
                text = ""
                warning = f"Text could not be extracted from PDF page {page_number}."
            pages.append(Page(
                pdf_page=page_number, text=text, character_count=len(text), warning=warning,
                extraction_method="text" if text else "none",
            ))
    except DocumentError:
        raise
    except (PyPdfError, ValueError, TypeError, KeyError, IndexError, OSError) as error:
        raise DocumentError("invalid_pdf", "This PDF could not be read. Try another file.") from error
    candidates = [page for page in pages if not page.text]
    ocr_unavailable = False
    if settings.ocr_enabled:
        if len(candidates) > settings.max_ocr_pages:
            raise DocumentError(
                "too_many_ocr_pages",
                f"At most {settings.max_ocr_pages} pages needing OCR are supported per PDF. "
                "Split the PDF into smaller files and upload again.",
            )
        for page in candidates:
            try:
                text = normalize_text(ocr.read_page(content, page.pdf_page - 1))
                if text:
                    page.text = text
                    page.character_count = len(text)
                    page.extraction_method = "ocr"
                    page.warning = (
                        f"PDF page {page.pdf_page} used OCR; check the text against the scan."
                    )
                else:
                    page.warning = f"PDF page {page.pdf_page}: OCR found no readable text."
            except OCRUnavailableError:
                ocr_unavailable = True
                page.warning = (
                    f"PDF page {page.pdf_page}: local OCR is unavailable. "
                    "Install requirements.txt and restart the backend."
                )
            except Exception:
                logger.exception("OCR failed on PDF page %s", page.pdf_page)
                page.warning = (
                    f"PDF page {page.pdf_page}: OCR could not read this page. "
                    "Try a clearer scan."
                )
    else:
        for page in candidates:
            page.warning = f"PDF page {page.pdf_page} has no embedded text; OCR is disabled."
    if not any(page.text for page in pages):
        if ocr_unavailable:
            raise DocumentError(
                "ocr_unavailable", "This PDF needs OCR, but the local engine is unavailable. "
                "Install requirements.txt and restart the backend.", 503,
            )
        raise DocumentError(
            "no_extractable_text",
            "No readable text was found. Try a clearer scan or a text PDF."
            if settings.ocr_enabled else
            "No embedded text was found. Enable DOCUASSIST_OCR_ENABLED for scanned PDFs.",
        )
    return pages


class DocumentService:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        self.lock = RLock()
        self.ocr = LocalOCR(settings)

    def initialize(self) -> None:
        self.storage.initialize()
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        # Recover a crash between renaming a file and committing its deletion.
        for document in self.storage.list_documents():
            original = self.file_path(document.id)
            tombstone = self.file_path(document.id, deleting=True)
            if tombstone.exists() and not original.exists():
                tombstone.replace(original)
        self.retry_cleanup()

    def file_path(self, document_id: str, deleting: bool = False):
        safe_id = UUID(document_id).hex
        suffix = ".pdf.deleting" if deleting else ".pdf"
        return self.settings.upload_dir / f"{safe_id}{suffix}"

    def retry_cleanup(self) -> list[str]:
        failed = []
        for document_id in self.storage.pending_cleanup():
            try:
                self.file_path(document_id, deleting=True).unlink(missing_ok=True)
                self.storage.complete_cleanup(document_id)
            except OSError:
                failed.append(document_id)
                logger.warning("PDF cleanup pending for %s", document_id)
        return failed

    def upload(self, filename: str | None, content: bytes) -> UploadResult:
        if len(content) > self.settings.max_upload_bytes:
            raise DocumentError(
                "file_too_large", f"PDFs may be at most {self.settings.max_upload_mb} MiB.", 413
            )
        name = display_filename(filename)
        digest = hashlib.sha256(content).hexdigest()
        with self.lock:
            existing = self.storage.find_by_hash(digest)
            if existing:
                return UploadResult(document=existing, duplicate=True)
            if len(self.storage.list_documents()) >= self.settings.max_documents:
                raise DocumentError(
                    "collection_full",
                    f"The collection holds at most {self.settings.max_documents} documents. "
                    "Delete a document before uploading another.", 409
                )
            pages = extract_pages(content, self.settings, self.ocr)
            warnings = [page.warning for page in pages if page.warning]
            document = Document(
                id=uuid4().hex, filename=name, sha256=digest, size_bytes=len(content),
                page_count=len(pages), text_page_count=sum(bool(page.text) for page in pages),
                status="extracted_with_warnings" if warnings else "extracted",
                warnings=warnings, created_at=datetime.now(timezone.utc).isoformat(),
            )
            original = self.file_path(document.id)
            try:
                with original.open("xb") as target:
                    target.write(content)
                self.storage.save_document(document, pages)
            except Exception:
                original.unlink(missing_ok=True)
                raise
            return UploadResult(document=document)

    def pages(self, document_id: str) -> DocumentPages:
        with self.lock:
            document = self.storage.get_document(document_id)
            if document is None:
                raise DocumentError("document_not_found", "This document was not found.", 404)
            return DocumentPages(document=document, pages=self.storage.get_pages(document_id))

    def reextract(self, document_id: str) -> DocumentPages:
        with self.lock:
            document = self.storage.get_document(document_id)
            if document is None:
                raise DocumentError("document_not_found", "This document was not found.", 404)
            if not self.settings.ocr_enabled:
                raise DocumentError(
                    "ocr_disabled", "Enable DOCUASSIST_OCR_ENABLED and restart the backend "
                    "to re-extract with OCR.", 409,
                )
            try:
                with self.file_path(document.id).open("rb") as source:
                    content = source.read(self.settings.max_upload_bytes + 1)
            except OSError as error:
                raise DocumentError(
                    "original_unavailable", "The stored original PDF could not be read. "
                    "Existing extracted text has been kept.", 409,
                ) from error
            if len(content) > self.settings.max_upload_bytes:
                raise DocumentError("file_too_large", "The stored PDF exceeds the current upload limit.", 413)
            if hashlib.sha256(content).hexdigest() != document.sha256:
                raise DocumentError(
                    "original_changed", "The stored PDF differs from the uploaded original. "
                    "Existing extracted text has been kept.", 409,
                )
            pages = extract_pages(content, self.settings, self.ocr)
            old_pages = self.storage.get_pages(document.id)
            new_text = {page.pdf_page: page.text for page in pages}
            if any(page.text and not new_text.get(page.pdf_page) for page in old_pages):
                raise DocumentError(
                    "reextraction_incomplete", "Re-extraction could not read a previously readable "
                    "page. Existing extracted text has been kept; try again after checking OCR.",
                )
            warnings = [page.warning for page in pages if page.warning]
            refreshed = document.model_copy(update={
                "page_count": len(pages), "text_page_count": sum(bool(page.text) for page in pages),
                "status": "extracted_with_warnings" if warnings else "extracted",
                "warnings": warnings,
            })
            self.storage.replace_extraction(refreshed, pages)
            return DocumentPages(document=refreshed, pages=pages)

    def delete(self, document_id: str) -> DeleteResult:
        with self.lock:
            document = self.storage.get_document(document_id)
            if document is None:
                raise DocumentError("document_not_found", "This document was not found.", 404)
            original = self.file_path(document.id)
            tombstone = self.file_path(document.id, deleting=True)
            renamed = False
            try:
                if original.exists():
                    original.replace(tombstone)
                    renamed = True
                self.storage.delete_document(document.id)
            except Exception:
                if renamed:
                    tombstone.replace(original)
                raise
            pending = self.retry_cleanup()
            warnings = (["Document records were removed; original-file cleanup will retry "
                         "on backend restart."] if document.id in pending else [])
            return DeleteResult(deleted=True, document_id=document.id, warnings=warnings)
