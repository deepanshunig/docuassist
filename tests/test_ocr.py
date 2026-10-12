import hashlib
import math
import sqlite3
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from backend.config import PROJECT_ROOT, Settings
from backend.main import create_app
from backend.ocr import LocalOCR, OCRUnavailableError, render_scale

TEXT_SAMPLE = PROJECT_ROOT / "sample_docs" / "fictional-vpn-guide.pdf"
SCAN_SAMPLE = PROJECT_ROOT / "sample_docs" / "fictional-scanned-vpn-guide.pdf"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", _env_file=None)


@pytest.fixture(scope="module")
def real_ocr():
    return LocalOCR(Settings(_env_file=None))


def pdf_bytes(*pages):
    writer = PdfWriter()
    for page in pages:
        writer.add_page(page)
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


def blank_page(width=200, height=200):
    return PdfWriter().add_blank_page(width=width, height=height)


def upload(client, content):
    return client.post("/documents", files={"file": ("scan.pdf", content, "application/pdf")})


def test_text_pdf_bypasses_ocr(settings, monkeypatch):
    app = create_app(settings)
    def unexpected_ocr(*args):
        raise AssertionError("Text PDF must bypass OCR")
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", unexpected_ocr)
    with TestClient(app) as client:
        response = upload(client, TEXT_SAMPLE.read_bytes())
        assert response.status_code == 201
        pages = client.get(f"/documents/{response.json()['document']['id']}/pages").json()["pages"]
        assert [page["extraction_method"] for page in pages] == ["text", "text"]
        assert app.state.document_service.ocr._engine is None


def test_real_image_only_pdf_ocr_duplicate_and_restart(settings, real_ocr):
    assert all(not page.extract_text().strip() for page in PdfReader(SCAN_SAMPLE).pages)
    app = create_app(settings)
    app.state.document_service.ocr = real_ocr
    with TestClient(app) as client:
        response = upload(client, SCAN_SAMPLE.read_bytes())
        assert response.status_code == 201
        document = response.json()["document"]
        assert document["text_page_count"] == document["page_count"] == 2
        pages = client.get(f"/documents/{document['id']}/pages").json()["pages"]
        assert [page["pdf_page"] for page in pages] == [1, 2]
        assert [page["extraction_method"] for page in pages] == ["ocr", "ocr"]
        assert "managed company devices" in pages[0]["text"]
        assert "install the approved Northstar Connect client" in pages[1]["text"]
        assert "multi-factor authentication (MFA)" in pages[1]["text"]
        assert all(page["warning"] and page["character_count"] == len(page["text"])
                   for page in pages)
        assert client.get("/health").json()["ocr"]["state"] == "ready"
        duplicate = upload(client, SCAN_SAMPLE.read_bytes())
        assert duplicate.status_code == 200 and duplicate.json()["duplicate"]
    with TestClient(create_app(settings)) as restarted:
        persisted = restarted.get(f"/documents/{document['id']}/pages").json()["pages"]
        assert persisted == pages
        assert restarted.delete(f"/documents/{document['id']}").status_code == 200
        assert restarted.get("/documents").json() == []
    assert list(settings.upload_dir.iterdir()) == []


def test_real_blank_page_has_no_hallucinated_text(real_ocr):
    assert real_ocr.read_page(pdf_bytes(blank_page()), 0) == ""


def test_mixed_pdf_fallback_keeps_pdf_page_numbers(settings, monkeypatch):
    app = create_app(settings)
    calls = []
    def fake_ocr(content, index):
        calls.append(index)
        return "" if index == 1 else "Recovered scan text"
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", fake_ocr)
    mixed = pdf_bytes(PdfReader(TEXT_SAMPLE).pages[0], blank_page(),
                      PdfReader(SCAN_SAMPLE).pages[1])
    with TestClient(app) as client:
        response = upload(client, mixed)
        assert response.status_code == 201
        pages = client.get(f"/documents/{response.json()['document']['id']}/pages").json()["pages"]
        assert calls == [1, 2]
        assert [page["pdf_page"] for page in pages] == [1, 2, 3]
        assert [page["extraction_method"] for page in pages] == ["text", "none", "ocr"]
        assert pages[1]["warning"] and pages[1]["text"] == ""
        assert pages[2]["text"] == "Recovered scan text"


def test_ocr_page_limit_rejects_before_running_ocr(settings, monkeypatch):
    settings.max_ocr_pages = 1
    app = create_app(settings)
    calls = []
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", lambda *args: calls.append(args))
    with TestClient(app) as client:
        response = upload(client, SCAN_SAMPLE.read_bytes())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "too_many_ocr_pages"
        assert calls == [] and client.get("/documents").json() == []
        assert list(settings.upload_dir.iterdir()) == []


@pytest.mark.parametrize("error", [OCRUnavailableError("missing engine"), RuntimeError("failed")])
def test_partial_ocr_failure_retains_readable_text(settings, monkeypatch, error):
    app = create_app(settings)
    def fail_ocr(*args):
        raise error
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", fail_ocr)
    content = pdf_bytes(PdfReader(TEXT_SAMPLE).pages[0], PdfReader(SCAN_SAMPLE).pages[1])
    with TestClient(app) as client:
        response = upload(client, content)
        assert response.status_code == 201
        document = response.json()["document"]
        assert document["status"] == "extracted_with_warnings"
        assert document["text_page_count"] == 1
        pages = client.get(f"/documents/{document['id']}/pages").json()["pages"]
        assert pages[0]["text"] and pages[1]["extraction_method"] == "none"
        assert "OCR" in pages[1]["warning"]


def test_engine_initialization_failure_is_actionable_and_text_upload_still_works(settings, monkeypatch):
    import rapidocr_onnxruntime
    calls = []
    def unavailable_engine(**kwargs):
        calls.append(kwargs)
        raise ImportError("simulated missing native runtime")
    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", unavailable_engine)
    with TestClient(create_app(settings)) as client:
        response = upload(client, SCAN_SAMPLE.read_bytes())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "ocr_unavailable"
        assert "requirements.txt" in response.json()["error"]["message"]
        assert len(calls) == 1 and client.get("/documents").json() == []
        assert list(settings.upload_dir.iterdir()) == []
        health = client.get("/health").json()
        assert health["ocr"]["state"] == "failed" and health["capabilities"]["ocr"] is False
        assert upload(client, TEXT_SAMPLE.read_bytes()).status_code == 201


def test_disabled_ocr_is_explicit_and_does_not_save_image_only_pdf(settings, monkeypatch):
    settings.ocr_enabled = False
    app = create_app(settings)
    calls = []
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", lambda *args: calls.append(args))
    with TestClient(app) as client:
        response = upload(client, SCAN_SAMPLE.read_bytes())
        assert response.status_code == 422
        assert "Enable DOCUASSIST_OCR_ENABLED" in response.json()["error"]["message"]
        assert calls == [] and client.get("/documents").json() == []
        assert client.get("/health").json()["ocr"]["state"] == "disabled"


def test_ocr_finding_no_text_rejects_entirely_empty_document(settings, monkeypatch):
    app = create_app(settings)
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", lambda *args: "")
    with TestClient(app) as client:
        response = upload(client, pdf_bytes(blank_page()))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "no_extractable_text"
        assert client.get("/documents").json() == []
        assert list(settings.upload_dir.iterdir()) == []


def test_oversized_page_render_is_bounded_before_ocr(settings):
    settings.ocr_max_pixels = 100_000
    ocr = LocalOCR(settings)
    observed = []
    def inspect_image(image):
        observed.append(image.shape)
        assert image.shape[0] * image.shape[1] <= settings.ocr_max_pixels
        return [[[], "test page", 1.0]], []
    ocr._engine = inspect_image
    assert ocr.read_page(pdf_bytes(blank_page(10000, 10000)), 0) == "test page"
    assert len(observed) == 1
    scale = render_scale(595.28, 841.89, 180, 4_000_000)
    assert math.ceil(595.28 * scale) * math.ceil(841.89 * scale) <= 4_000_000


def test_version_one_database_migrates_without_losing_pages_or_originals(settings):
    settings.data_dir.mkdir(parents=True)
    settings.upload_dir.mkdir()
    identifier = "a" * 32
    original = settings.upload_dir / f"{identifier}.pdf"
    original.write_bytes(TEXT_SAMPLE.read_bytes())
    with sqlite3.connect(settings.database_path) as connection:
        connection.executescript("""
            CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE, size_bytes INTEGER NOT NULL, page_count INTEGER NOT NULL,
                text_page_count INTEGER NOT NULL, status TEXT NOT NULL, warnings TEXT NOT NULL,
                created_at TEXT NOT NULL);
            CREATE TABLE pages (document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                pdf_page INTEGER NOT NULL, text TEXT NOT NULL, character_count INTEGER NOT NULL,
                warning TEXT, PRIMARY KEY (document_id, pdf_page));
            PRAGMA user_version = 1;
        """)
        connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            identifier, "legacy.pdf", hashlib.sha256(original.read_bytes()).hexdigest(),
            original.stat().st_size, 2, 1, "extracted_with_warnings", "[]", "2026-01-01",
        ))
        connection.executemany("INSERT INTO pages VALUES (?, ?, ?, ?, ?)", [
            (identifier, 1, "Existing text", 13, None),
            (identifier, 2, "", 0, "Original warning"),
        ])
    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            pages = client.get(f"/documents/{identifier}/pages").json()["pages"]
            assert [page["extraction_method"] for page in pages] == ["text", "none"]
            assert pages[0]["text"] == "Existing text" and pages[1]["warning"] == "Original warning"
            assert upload(client, TEXT_SAMPLE.read_bytes()).json()["duplicate"]
            assert original.read_bytes() == TEXT_SAMPLE.read_bytes()
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    with TestClient(create_app(settings)) as client:
        other = TEXT_SAMPLE.parent / "fictional-password-guide.pdf"
        assert upload(client, other.read_bytes()).status_code == 201


def test_reextract_existing_mixed_pdf_runs_real_ocr_and_preserves_identity(settings, real_ocr):
    settings.ocr_enabled = False
    app = create_app(settings)
    content = pdf_bytes(PdfReader(TEXT_SAMPLE).pages[0], PdfReader(SCAN_SAMPLE).pages[1])
    with TestClient(app) as client:
        initial = upload(client, content).json()["document"]
        assert initial["text_page_count"] == 1
        original = settings.upload_dir / f"{initial['id']}.pdf"
        before = original.read_bytes()
        settings.ocr_enabled = True
        app.state.document_service.ocr = real_ocr
        response = client.post(f"/documents/{initial['id']}/reextract")
        assert response.status_code == 200
        refreshed = response.json()
        assert refreshed["document"]["text_page_count"] == 2
        for key in ("id", "filename", "sha256", "size_bytes", "created_at"):
            assert refreshed["document"][key] == initial[key]
        assert [page["extraction_method"] for page in refreshed["pages"]] == ["text", "ocr"]
        assert "install the approved Northstar Connect client" in refreshed["pages"][1]["text"]
        assert original.read_bytes() == before and len(client.get("/documents").json()) == 1
        duplicate = upload(client, content).json()
        assert duplicate["duplicate"] and duplicate["document"]["text_page_count"] == 2
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/documents/{initial['id']}/pages").json() == refreshed


def test_reextract_transaction_failure_restores_previous_pages_and_metadata(settings, monkeypatch):
    settings.ocr_enabled = False
    app = create_app(settings)
    content = pdf_bytes(PdfReader(TEXT_SAMPLE).pages[0], PdfReader(SCAN_SAMPLE).pages[1])
    with TestClient(app, raise_server_exceptions=False) as client:
        initial = upload(client, content).json()["document"]
        path = f"/documents/{initial['id']}/pages"
        before = client.get(path).json()
        settings.ocr_enabled = True
        monkeypatch.setattr(app.state.document_service.ocr, "read_page", lambda *args: "New OCR text")
        # Fail after the metadata update, page deletion and first page insertion.
        with sqlite3.connect(settings.database_path) as connection:
            connection.executescript("""
                CREATE TRIGGER fail_second_page BEFORE INSERT ON pages
                WHEN NEW.pdf_page = 2 BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END;
            """)
        response = client.post(f"/documents/{initial['id']}/reextract")
        assert response.status_code == 500
        assert client.get(path).json() == before
        assert (settings.upload_dir / f"{initial['id']}.pdf").read_bytes() == content


@pytest.mark.parametrize("problem,code", [
    ("missing", "original_unavailable"), ("changed", "original_changed"),
])
def test_reextract_original_error_preserves_saved_text(settings, problem, code):
    with TestClient(create_app(settings)) as client:
        initial = upload(client, TEXT_SAMPLE.read_bytes()).json()["document"]
        path = f"/documents/{initial['id']}/pages"
        before = client.get(path).json()
        original = settings.upload_dir / f"{initial['id']}.pdf"
        if problem == "missing":
            original.unlink()
        else:
            original.write_bytes(b"changed test original")
        response = client.post(f"/documents/{initial['id']}/reextract")
        assert response.status_code == 409 and response.json()["error"]["code"] == code
        assert client.get(path).json() == before


def test_reextract_disabled_ocr_and_unknown_document_are_actionable(settings):
    settings.ocr_enabled = False
    with TestClient(create_app(settings)) as client:
        assert client.post("/documents/unknown/reextract").status_code == 404
        initial = upload(client, TEXT_SAMPLE.read_bytes()).json()["document"]
        response = client.post(f"/documents/{initial['id']}/reextract")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ocr_disabled"


def test_reextract_does_not_erase_previously_recognized_ocr_text(settings, monkeypatch):
    app = create_app(settings)
    monkeypatch.setattr(app.state.document_service.ocr, "read_page", lambda *args: "Saved OCR text")
    content = pdf_bytes(PdfReader(TEXT_SAMPLE).pages[0], PdfReader(SCAN_SAMPLE).pages[1])
    with TestClient(app) as client:
        initial = upload(client, content).json()["document"]
        path = f"/documents/{initial['id']}/pages"
        before = client.get(path).json()
        def unavailable_ocr(*args):
            raise OCRUnavailableError("engine unavailable")
        monkeypatch.setattr(app.state.document_service.ocr, "read_page", unavailable_ocr)
        response = client.post(f"/documents/{initial['id']}/reextract")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "reextraction_incomplete"
        assert client.get(path).json() == before
