"""Deterministic lifecycle/citation checks; real-model checks are clearly separate."""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from backend.answering import validate_answer
from backend.chunking import split_page
from backend.config import PROJECT_ROOT, Settings
from backend.embeddings import EmbeddingEngine
from backend.errors import DocumentError
from backend.main import create_app
from backend.llm import LocalGenerator
from backend.schemas import Source

VPN = PROJECT_ROOT / "sample_docs" / "fictional-vpn-guide.pdf"
PASSWORD = PROJECT_ROOT / "sample_docs" / "fictional-password-guide.pdf"
SCANNED = PROJECT_ROOT / "sample_docs" / "fictional-scanned-vpn-guide.pdf"


class ControlledEmbedding:
    """Not a semantic model; known vectors only test collection plumbing."""
    dimension = 4
    model_id = "controlled-test-embedding"

    def __init__(self, settings):
        self.fingerprint = hashlib.sha256(
            f"test:{settings.chunk_tokens}:{settings.chunk_overlap}".encode()).hexdigest()
        self.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.calls = 0
        self.fail = False

    def load(self):
        pass

    def encode(self, texts):
        self.calls += 1
        if self.fail:
            raise DocumentError("embedding_failed", "Controlled test failure", 503)
        vectors = []
        for text in texts:
            if len(self.tokenizer.encode(text).ids) > 256:
                raise DocumentError("question_too_long", "Ask a shorter question")
            text = text.lower()
            vector = np.array([sum(text.count(w) for w in ("vpn", "northstar", "connect")),
                sum(text.count(w) for w in ("password", "identity", "reset")),
                sum(text.count(w) for w in ("printer", "device")), 0.01], dtype=np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors, dtype=np.float32).reshape(-1, self.dimension)


class ControlledGenerator:
    model = "controlled-test-generator"

    def __init__(self):
        self.calls = 0
        self.responses = []
        self.error = None
        self.messages = None

    def status(self):
        return {"state": "ready", "provider": "controlled", "model": self.model}

    def count_prompt_tokens(self, messages):
        return 800

    def generate(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({"status": "answered", "answer": "Install Northstar Connect and enroll in MFA. [S1]",
                           "citations": ["S1"]})


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", ocr_enabled=False, _env_file=None)


@pytest.fixture
def app(settings):
    return create_app(settings, ControlledEmbedding(settings), ControlledGenerator())


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def upload(client, path=VPN):
    response = client.post("/documents", files={"file": (path.name, path.read_bytes(), "application/pdf")})
    assert response.status_code in (200, 201), response.text
    return response.json()


def search(client, **kwargs):
    return client.post("/search", json={"question": "What must I do before connecting to the VPN?", **kwargs})


def test_token_windows_preserve_text_and_have_bounded_overlap(settings):
    engine = ControlledEmbedding(settings)
    text = " ".join(f"word{i}" for i in range(600))
    chunks = list(split_page(text, engine.tokenizer, size=100, overlap=20))
    assert len(chunks) > 1
    assert all(count <= 102 for _, count in chunks)
    original = set(text.split())
    recovered = {word for chunk, _ in chunks for word in chunk.split()}
    assert original == recovered
    assert set(chunks[0][0].split()) & set(chunks[1][0].split())


def test_upload_search_and_cited_answer(client, app):
    result = upload(client)
    assert result["indexed"] and result["indexing_warning"] is None
    hit = search(client).json()
    assert hit["sources"][0]["document_id"] == result["document"]["id"]
    assert hit["sources"][0]["pdf_page"] == 2
    answer = client.post("/ask", json={"question": "What must I do before connecting to the VPN?"}).json()
    assert answer["status"] == "answered"
    assert answer["sources"][0]["chunk_id"] in {s["chunk_id"] for s in answer["retrieved_sources"]}
    assert answer["sources"][0]["filename"] == VPN.name
    assert answer["model"] == "controlled-test-generator"
    prompt = json.loads(app.state.answering_service.generator.messages[1]["content"])
    assert prompt["sources"] and "untrusted document" in app.state.answering_service.generator.messages[0]["content"]


def test_duplicate_does_not_reembed_or_duplicate_vectors(client, app):
    first = upload(client)
    count = app.state.retrieval_service.status()["chunk_count"]
    calls = app.state.retrieval_service.engine.calls
    second = upload(client)
    assert second["duplicate"] and second["indexed"]
    assert first["document"]["id"] == second["document"]["id"]
    assert app.state.retrieval_service.status()["chunk_count"] == count
    assert app.state.retrieval_service.engine.calls == calls


def test_restart_reuses_vectors_and_deletion_stays_deleted(settings):
    first_app = create_app(settings, ControlledEmbedding(settings), ControlledGenerator())
    with TestClient(first_app) as first:
        identifier = upload(first)["document"]["id"]
        original_ids = {s["chunk_id"] for s in search(first).json()["sources"]}
    fresh_engine = ControlledEmbedding(settings)
    with TestClient(create_app(settings, fresh_engine, ControlledGenerator())) as restarted:
        assert fresh_engine.calls == 0
        assert original_ids == {s["chunk_id"] for s in search(restarted).json()["sources"]}
        assert restarted.delete(f"/documents/{identifier}").status_code == 200
        assert search(restarted).json()["sources"] == []
    with TestClient(create_app(settings, ControlledEmbedding(settings), ControlledGenerator())) as final:
        assert final.get("/health").json()["rag"]["retrieval"]["chunk_count"] == 0
        assert search(final).json()["sources"] == []
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM document_index").fetchone()[0] == 0


def test_search_filters_documents_before_top_k(client):
    vpn = upload(client)["document"]["id"]
    password = upload(client, PASSWORD)["document"]["id"]
    result = search(client, document_ids=[password], k=1).json()
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == password
    assert result["sources"][0]["document_id"] != vpn
    assert search(client, document_ids=["missing"]).status_code == 404


def test_empty_collection_does_not_call_generator(client, app):
    answer = client.post("/ask", json={"question": "How do I connect?"}).json()
    assert answer["status"] == "insufficient_evidence" and answer["sources"] == []
    assert answer["model"] is None
    assert app.state.answering_service.generator.calls == 0


@pytest.mark.parametrize("question", ["I cannot get access. What should I do?", "It failed. Which steps should I follow?"])
def test_vague_standalone_question_requests_clarification(client, app, question):
    upload(client)
    answer = client.post("/ask", json={"question": question}).json()
    assert answer["status"] == "clarification"
    assert answer["sources"] == []
    assert app.state.answering_service.generator.calls == 0


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_invalid_question_is_rejected(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_long_embedding_question_is_rejected_without_truncation(client):
    upload(client)
    result = client.post("/search", json={"question": "vpn " * 300})
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "question_too_long"


def test_unknown_citations_retry_once_then_fail(client, app):
    upload(client)
    generator = app.state.answering_service.generator
    bad = json.dumps({"status": "answered", "answer": "Invented answer [S99]", "citations": ["S99"]})
    generator.responses = [bad, bad]
    response = client.post("/ask", json={"question": "How do I connect to VPN?"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_citations"
    assert generator.calls == 2


def test_unknown_id_inside_grouped_citation_is_rejected(client, app):
    upload(client)
    generator = app.state.answering_service.generator
    bad = json.dumps({"status": "answered", "answer": "An answer [S1, S99]", "citations": ["S1"]})
    generator.responses = [bad, bad]
    response = client.post("/ask", json={"question": "How do I connect to VPN?"})
    assert response.status_code == 502
    assert generator.calls == 2


def test_malformed_response_can_recover_on_one_retry(client, app):
    upload(client)
    app.state.answering_service.generator.responses = ["not JSON"]
    result = client.post("/ask", json={"question": "How do I connect to VPN?"})
    assert result.status_code == 200
    assert app.state.answering_service.generator.calls == 2


def test_valid_structured_citations_are_rendered_if_model_omits_markers(client, app):
    upload(client)
    app.state.answering_service.generator.responses = [json.dumps({"status": "answered",
        "answer": "Install the client and enroll in MFA.", "citations": ["S1"]})]
    response = client.post("/ask", json={"question": "How do I connect to VPN?"})
    assert response.status_code == 200
    assert response.json()["answer"].endswith("[S1]")
    assert response.json()["sources"][0]["source_id"] == "S1"


@pytest.mark.parametrize("status", ["answered", "insufficient_evidence"])
def test_explicit_refusal_is_not_presented_as_a_supported_answer(client, app, status):
    upload(client)
    app.state.answering_service.generator.responses = [json.dumps({"status": status,
        "answer": "The passage does not provide information about laptop replacements. [S1]",
        "citations": ["S1"]})]
    response = client.post("/ask", json={"question": "How often are VPN laptops replaced?"})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["sources"] == []


def test_provider_timeout_is_not_reported_as_missing_evidence(client, app):
    upload(client)
    app.state.answering_service.generator.error = DocumentError("generator_timeout", "Timed out", 504)
    response = client.post("/ask", json={"question": "How do I connect to VPN?"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "generator_timeout"


def test_failed_indexing_preserves_previous_collection(client, app):
    upload(client)
    previous = app.state.retrieval_service.status()
    app.state.retrieval_service.engine.fail = True
    response = client.post("/documents", files={"file": (PASSWORD.name, PASSWORD.read_bytes(), "application/pdf")})
    assert response.status_code == 503
    assert len(client.get("/documents").json()) == 1
    assert app.state.retrieval_service.status() == previous


def test_database_failure_keeps_vector_mapping_and_original(client, app, settings):
    upload(client)
    previous = app.state.retrieval_service.status()
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("CREATE TRIGGER fail_vector BEFORE INSERT ON chunks BEGIN "
                           "SELECT RAISE(ABORT, 'controlled vector failure'); END")
    with TestClient(app, raise_server_exceptions=False) as failing:
        response = failing.post("/documents", files={"file": (PASSWORD.name, PASSWORD.read_bytes(), "application/pdf")})
        assert response.status_code == 500
    assert app.state.retrieval_service.status() == previous
    assert len(list(settings.upload_dir.glob("*.pdf"))) == 1
    assert len(client.get("/documents").json()) == 1


def test_configuration_change_reindexes_existing_documents(settings):
    with TestClient(create_app(settings, ControlledEmbedding(settings), ControlledGenerator())) as first:
        identifier = upload(first)["document"]["id"]
    changed = settings.model_copy(update={"chunk_tokens": 80, "chunk_overlap": 10})
    engine = ControlledEmbedding(changed)
    with TestClient(create_app(changed, engine, ControlledGenerator())) as restarted:
        assert engine.calls > 0
        assert search(restarted).json()["sources"][0]["document_id"] == identifier
        assert all(c.fingerprint == engine.fingerprint for c in restarted.app.state.document_service.storage.list_chunks())


def test_reextract_refreshes_search_and_rolls_back_vectors_on_database_failure(settings):
    settings.ocr_enabled = True
    app = create_app(settings, ControlledEmbedding(settings), ControlledGenerator())
    app.state.document_service.ocr.read_page = lambda *args: "Original VPN prerequisite text."
    with TestClient(app, raise_server_exceptions=False) as client:
        identifier = upload(client, SCANNED)["document"]["id"]
        original_chunks = app.state.document_service.storage.list_chunks()
        app.state.document_service.ocr.read_page = lambda *args: "Updated VPN client and MFA prerequisites."
        refreshed = client.post(f"/documents/{identifier}/reextract")
        assert refreshed.status_code == 200
        updated_chunks = app.state.document_service.storage.list_chunks()
        assert {c.id for c in updated_chunks}.isdisjoint({c.id for c in original_chunks})
        assert all("Updated" in s["passage"] for s in search(client).json()["sources"])
        previous = app.state.retrieval_service.status()
        with sqlite3.connect(settings.database_path) as connection:
            connection.execute("CREATE TRIGGER fail_vector BEFORE INSERT ON chunks BEGIN "
                               "SELECT RAISE(ABORT, 'controlled vector failure'); END")
        app.state.document_service.ocr.read_page = lambda *args: "Another VPN update."
        assert client.post(f"/documents/{identifier}/reextract").status_code == 500
        assert app.state.document_service.storage.list_chunks() == updated_chunks
        assert app.state.retrieval_service.status() == previous
        assert all("Updated" in s["passage"] for s in search(client).json()["sources"])


def test_answer_and_delete_share_a_consistent_collection_lock(client, app):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    identifier = upload(client)["document"]["id"]
    started, release = Event(), Event()
    original = app.state.answering_service.generator.generate

    def blocked(messages):
        started.set()
        assert release.wait(10)
        return original(messages)

    app.state.answering_service.generator.generate = blocked
    with ThreadPoolExecutor(max_workers=2) as pool:
        answering = pool.submit(client.post, "/ask", json={"question": "How do I connect to VPN?"})
        assert started.wait(5)
        deleting = pool.submit(client.delete, f"/documents/{identifier}")
        try:
            assert not deleting.done()
        finally:
            release.set()
        answer = answering.result(timeout=10)
        deletion = deleting.result(timeout=10)
    assert answer.status_code == deletion.status_code == 200
    assert answer.json()["sources"][0]["document_id"] == identifier
    assert search(client).json()["sources"] == []


def test_context_budget_reduces_evidence_before_generation(client, app):
    upload(client)
    generator = app.state.answering_service.generator
    generator.count_prompt_tokens = lambda messages: 4000 if len(json.loads(messages[1]["content"])["sources"]) > 1 else 1000
    response = client.post("/ask", json={"question": "How do I connect to VPN?"})
    assert response.status_code == 200
    assert len(response.json()["retrieved_sources"]) == 1


def test_real_minilm_is_normalized_and_finds_vpn_prerequisites():
    settings = Settings(_env_file=None)
    if not (settings.model_dir / "manifest.json").exists():
        pytest.skip("Run scripts/setup_models.py for the real embedding integration check")
    engine = EmbeddingEngine(settings)
    engine.load()
    vectors = engine.encode(["What is required before using the company VPN?",
        "Install the Northstar Connect client and enroll your account in MFA before connecting.",
        "The cafeteria serves lunch at noon."])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    with pytest.raises(DocumentError, match="256-token"):
        engine.encode(["vpn " * 300])


def test_malformed_model_server_health_does_not_break_backend_health(settings, monkeypatch):
    import httpx
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs))
    assert LocalGenerator(settings).status()["state"] == "unavailable"


def test_real_adapter_timeout_preserves_provider_error_code(settings, monkeypatch):
    import httpx
    original_client = httpx.Client

    def timeout(request):
        raise httpx.ReadTimeout("Controlled timeout", request=request)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(timeout), **kwargs))
    with pytest.raises(DocumentError) as caught:
        LocalGenerator(settings).request("/completion", {"prompt": "test"})
    assert caught.value.status_code == 504 and caught.value.code == "generator_timeout"
