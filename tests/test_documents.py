import sqlite3
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from backend.config import PROJECT_ROOT, Settings
from backend.main import create_app

SAMPLE = PROJECT_ROOT / "sample_docs" / "fictional-vpn-guide.pdf"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", ocr_enabled=False, _env_file=None)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def upload(client, content=None, name="guide.pdf"):
    return client.post("/documents", files={"file": (
        name, SAMPLE.read_bytes() if content is None else content, "application/pdf"
    )})


def pdf_bytes(writer):
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


def test_upload_preserves_page_order_and_source_text(client, settings):
    response = upload(client)
    assert response.status_code == 201
    document = response.json()["document"]
    assert document["page_count"] == document["text_page_count"] == 2
    assert document["status"] == "extracted"
    result = client.get(f"/documents/{document['id']}/pages").json()
    assert [page["pdf_page"] for page in result["pages"]] == [1, 2]
    assert "Purpose and access" in result["pages"][0]["text"]
    assert "enroll your corporate account" in " ".join(result["pages"][1]["text"].split())
    assert result["pages"][1]["character_count"] == len(result["pages"][1]["text"])
    assert len(list(settings.upload_dir.glob("*.pdf"))) == 1


def test_exact_duplicate_reuses_existing_document(client, settings):
    first = upload(client).json()
    second_response = upload(client, name="renamed.pdf")
    second = second_response.json()
    assert second_response.status_code == 200
    assert second["duplicate"] is True
    assert second["document"]["id"] == first["document"]["id"]
    assert len(client.get("/documents").json()) == 1
    assert len(list(settings.upload_dir.glob("*.pdf"))) == 1


def test_uploaded_filename_cannot_choose_storage_location(client, settings):
    document = upload(client, name="../../outside.pdf").json()["document"]
    assert document["filename"] == "outside.pdf"
    assert (settings.upload_dir / f"{document['id']}.pdf").is_file()
    assert not (settings.data_dir.parent / "outside.pdf").exists()


@pytest.mark.parametrize("content,code", [
    (b"", "empty_file"),
    (b"not a PDF", "invalid_pdf"),
    (b"%PDF-1.7\nmalformed", "invalid_pdf"),
])
def test_invalid_upload_creates_no_records_or_files(client, settings, content, code):
    response = upload(client, content=content)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert client.get("/documents").json() == []
    assert list(settings.upload_dir.iterdir()) == []


def test_non_pdf_extension_is_rejected(client):
    response = upload(client, name="guide.txt")
    assert response.status_code == 415


def test_encrypted_pdf_is_rejected(client):
    writer = PdfWriter()
    writer.append(PdfReader(SAMPLE))
    writer.encrypt("example-password")
    response = upload(client, content=pdf_bytes(writer))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "encrypted_pdf"


def test_blank_pdf_reports_missing_text(client):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    response = upload(client, content=pdf_bytes(writer))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_extractable_text"


def test_partial_extraction_keeps_blank_page_number_and_warning(client):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_page(PdfReader(SAMPLE).pages[1])
    response = upload(client, content=pdf_bytes(writer))
    assert response.status_code == 201
    document = response.json()["document"]
    assert document["status"] == "extracted_with_warnings"
    assert document["text_page_count"] == 1
    pages = client.get(f"/documents/{document['id']}/pages").json()["pages"]
    assert pages[0]["warning"] and pages[0]["text"] == ""
    assert pages[1]["pdf_page"] == 2
    assert "Prerequisites" in pages[1]["text"]


def test_size_and_page_limits_are_enforced(settings):
    settings.max_upload_mb = 1
    settings.max_pages = 1
    with TestClient(create_app(settings)) as client:
        large = upload(client, content=b"%PDF-" + b"x" * (1024 * 1024))
        assert large.status_code == 413
        pages = upload(client)
        assert pages.status_code == 422
        assert pages.json()["error"]["code"] == "too_many_pages"
        assert client.get("/documents").json() == []


def test_collection_limit_allows_duplicate_but_rejects_new_file(settings):
    settings.max_documents = 1
    with TestClient(create_app(settings)) as client:
        assert upload(client).status_code == 201
        assert upload(client).status_code == 200
        other = (SAMPLE.parent / "fictional-password-guide.pdf").read_bytes()
        response = upload(client, content=other)
        assert response.status_code == 409


def test_documents_survive_restart_and_deleted_pages_stay_deleted(settings):
    with TestClient(create_app(settings)) as first:
        document_id = upload(first).json()["document"]["id"]
    with TestClient(create_app(settings)) as restarted:
        assert len(restarted.get("/documents").json()) == 1
        assert len(restarted.get(f"/documents/{document_id}/pages").json()["pages"]) == 2
        assert restarted.delete(f"/documents/{document_id}").status_code == 200
        assert restarted.get(f"/documents/{document_id}/pages").status_code == 404
        assert restarted.delete(f"/documents/{document_id}").status_code == 404
    with TestClient(create_app(settings)) as final:
        assert final.get("/documents").json() == []
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM pages").fetchone()[0] == 0
    assert list(settings.upload_dir.iterdir()) == []


def test_database_failure_rolls_back_file_and_pages(settings, monkeypatch):
    application = create_app(settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        def fail_save(*args):
            raise sqlite3.OperationalError("simulated database failure")
        monkeypatch.setattr(application.state.document_service.storage, "save_document", fail_save)
        assert upload(client).status_code == 500
        assert client.get("/documents").json() == []
        assert list(settings.upload_dir.iterdir()) == []


def test_restart_recovers_file_rename_before_deletion_commit(settings):
    application = create_app(settings)
    with TestClient(application) as client:
        identifier = upload(client).json()["document"]["id"]
        service = application.state.document_service
        service.file_path(identifier).replace(service.file_path(identifier, deleting=True))
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/documents/{identifier}/pages").status_code == 200
        assert (settings.upload_dir / f"{identifier}.pdf").exists()


def test_pending_file_cleanup_is_retried_on_restart(settings, monkeypatch):
    with TestClient(create_app(settings)) as client:
        identifier = upload(client).json()["document"]["id"]
        original_unlink = Path.unlink
        def block_cleanup(path, *args, **kwargs):
            if path.name.endswith(".deleting"):
                raise PermissionError("simulated locked PDF")
            return original_unlink(path, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", block_cleanup)
            response = client.delete(f"/documents/{identifier}")
            assert response.status_code == 200
            assert response.json()["warnings"]
            assert client.get("/documents").json() == []
    with TestClient(create_app(settings)):
        assert list(settings.upload_dir.iterdir()) == []
