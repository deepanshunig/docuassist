import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.schemas import Document, Page


class Storage:
    """SQLite is canonical; no page numbering is reconstructed from filenames."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    text_page_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    pdf_page INTEGER NOT NULL CHECK (pdf_page >= 1),
                    text TEXT NOT NULL,
                    character_count INTEGER NOT NULL,
                    warning TEXT,
                    extraction_method TEXT NOT NULL DEFAULT 'none',
                    PRIMARY KEY (document_id, pdf_page)
                );
                CREATE TABLE IF NOT EXISTS pending_cleanup (
                    document_id TEXT PRIMARY KEY
                );
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(pages)")}
            if "extraction_method" not in columns:
                connection.execute(
                    "ALTER TABLE pages ADD COLUMN extraction_method TEXT NOT NULL DEFAULT 'none'"
                )
                connection.execute(
                    "UPDATE pages SET extraction_method = CASE "
                    "WHEN length(trim(text)) > 0 THEN 'text' ELSE 'none' END"
                )
            if connection.execute("PRAGMA user_version").fetchone()[0] < 2:
                connection.execute("PRAGMA user_version = 2")

    @staticmethod
    def document_from_row(row: sqlite3.Row) -> Document:
        values = dict(row)
        values["warnings"] = json.loads(values["warnings"])
        return Document.model_validate(values)

    def list_documents(self) -> list[Document]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self.document_from_row(row) for row in rows]

    def get_document(self, document_id: str) -> Document | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self.document_from_row(row) if row else None

    def find_by_hash(self, sha256: str) -> Document | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self.document_from_row(row) if row else None

    def get_pages(self, document_id: str) -> list[Page]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT pdf_page, text, character_count, warning, extraction_method FROM pages "
                "WHERE document_id = ? ORDER BY pdf_page", (document_id,)
            ).fetchall()
        return [Page.model_validate(dict(row)) for row in rows]

    def save_document(self, document: Document, pages: list[Page]) -> None:
        values = document.model_dump()
        values["warnings"] = json.dumps(document.warnings)
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO documents VALUES "
                "(:id, :filename, :sha256, :size_bytes, :page_count, "
                ":text_page_count, :status, :warnings, :created_at)", values
            )
            connection.executemany(
                "INSERT INTO pages (document_id, pdf_page, text, character_count, warning, "
                "extraction_method) VALUES (?, ?, ?, ?, ?, ?)",
                [(document.id, page.pdf_page, page.text, page.character_count, page.warning,
                  page.extraction_method)
                 for page in pages],
            )

    def delete_document(self, document_id: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO pending_cleanup VALUES (?)", (document_id,)
            )
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def replace_extraction(self, document: Document, pages: list[Page]) -> None:
        """Commit refreshed metadata and all pages together; failures retain old results."""
        values = document.model_dump()
        values["warnings"] = json.dumps(document.warnings)
        with self.connection() as connection:
            connection.execute(
                "UPDATE documents SET page_count = :page_count, text_page_count = :text_page_count, "
                "status = :status, warnings = :warnings WHERE id = :id", values,
            )
            connection.execute("DELETE FROM pages WHERE document_id = ?", (document.id,))
            connection.executemany(
                "INSERT INTO pages (document_id, pdf_page, text, character_count, warning, "
                "extraction_method) VALUES (?, ?, ?, ?, ?, ?)",
                [(document.id, page.pdf_page, page.text, page.character_count, page.warning,
                  page.extraction_method) for page in pages],
            )

    def pending_cleanup(self) -> list[str]:
        with self.connection() as connection:
            rows = connection.execute("SELECT document_id FROM pending_cleanup").fetchall()
        return [row["document_id"] for row in rows]

    def complete_cleanup(self, document_id: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM pending_cleanup WHERE document_id = ?", (document_id,)
            )
