"""One immutable FAISS index/mapping pair, swapped under the document lock."""
import logging
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from backend.chunking import create_chunks, source_digest
from backend.embeddings import EmbeddingEngine
from backend.errors import DocumentError
from backend.schemas import ChunkRecord, SearchResult, Source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Snapshot:
    index: object
    records: tuple[ChunkRecord, ...]


@dataclass(frozen=True)
class IndexPlan:
    chunks: list[ChunkRecord]
    digest: str
    snapshot: Snapshot


class RetrievalService:
    def __init__(self, settings, storage, lock, engine=None):
        self.settings = settings
        self.storage = storage
        self.lock = lock
        self.engine = engine or EmbeddingEngine(settings)
        self.snapshot = None
        self.ready = False
        self.state = "not_loaded"
        self.message = None

    def build_snapshot(self, records):
        import faiss
        records = tuple(sorted(records, key=lambda item: item.id))
        index = faiss.IndexFlatIP(self.engine.dimension)
        if records:
            if any(c.dimension != self.engine.dimension or c.fingerprint != self.engine.fingerprint
                   or len(c.vector) != self.engine.dimension * 4 for c in records):
                raise ValueError("Stored vector configuration mismatch")
            vectors = np.stack([np.frombuffer(c.vector, dtype="<f4") for c in records])
            if not np.isfinite(vectors).all() or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1, atol=0.001
            ):
                raise ValueError("Stored vectors must be finite and normalized")
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        if index.ntotal != len(records):
            raise ValueError("Vector-to-source mapping mismatch")
        return Snapshot(index, records)

    def initialize(self):
        if not self.settings.rag_enabled:
            self.state = "disabled"
            return
        try:
            with self.lock:
                self.engine.load()
                metadata = self.storage.index_metadata()
                stored = defaultdict(list)
                for chunk in self.storage.list_chunks():
                    stored[chunk.document_id].append(chunk)
                replacements, records = [], []
                for document in self.storage.list_documents():
                    pages = self.storage.get_pages(document.id)
                    digest = source_digest(pages)
                    previous = metadata.get(document.id, {})
                    chunks = stored[document.id]
                    valid = (previous.get("fingerprint") == self.engine.fingerprint
                             and previous.get("source_digest") == digest
                             and previous.get("chunk_count") == len(chunks) and bool(chunks))
                    if valid:
                        try:
                            self.build_snapshot(chunks)
                        except ValueError:
                            valid = False
                    if not valid:
                        chunks = create_chunks(document, pages, self.engine, self.settings)
                        replacements.append((document.id, chunks, digest))
                    records.extend(chunks)
                snapshot = self.build_snapshot(records)
                if replacements:
                    self.storage.save_indexes(replacements)
                self.snapshot = snapshot
                self.ready = True
                self.state = "ready"
                self.message = None
        except Exception:
            logger.exception("Local retrieval initialization failed")
            self.ready = False
            self.state = "unavailable"
            self.message = "Run python scripts/setup_models.py and restart the backend. "
            self.message += "If models are already installed, check the backend log."

    def status(self):
        with self.lock:
            records = self.snapshot.records if self.snapshot else ()
            return {"state": self.state, "message": self.message, "model": self.engine.model_id,
                    "revision": self.settings.embedding_revision, "chunk_count": len(records),
                    "indexed_document_count": len({c.document_id for c in records}),
                    "collection_version": self.storage.collection_version()}

    def prepare(self, document, pages):
        if not self.ready:
            return None
        try:
            chunks = create_chunks(document, pages, self.engine, self.settings)
            snapshot = self.build_snapshot([c for c in self.snapshot.records
                if c.document_id != document.id] + chunks)
            return IndexPlan(chunks, source_digest(pages), snapshot)
        except DocumentError:
            raise
        except Exception as error:
            raise DocumentError("indexing_failed", "The PDF could not be indexed. "
                "Previous document and search data have been kept. Check the backend log.", 503) from error

    def prepare_delete(self, document_id):
        if not self.ready:
            return None
        return self.build_snapshot([c for c in self.snapshot.records if c.document_id != document_id])

    def activate(self, snapshot):
        if snapshot is not None:
            self.snapshot = snapshot

    def is_indexed(self, document_id):
        return self.ready and any(c.document_id == document_id for c in self.snapshot.records)

    def search(self, request):
        started = perf_counter()
        with self.lock:
            if not self.ready:
                raise DocumentError("retrieval_unavailable", "Semantic search is not ready. "
                    "Run python scripts/setup_models.py and restart the backend.", 503)
            snapshot = self.snapshot
            if request.document_ids is not None:
                allowed = set(request.document_ids)
                existing = {d.id for d in self.storage.list_documents()}
                if not allowed <= existing:
                    raise DocumentError("document_not_found", "A selected document no longer exists. Refresh the collection.", 404)
                snapshot = self.build_snapshot([c for c in snapshot.records if c.document_id in allowed])
            sources = []
            if snapshot.records:
                query = self.engine.encode([request.question])
                k = min(request.k or self.settings.retrieval_k, len(snapshot.records))
                scores, positions = snapshot.index.search(query, k)
                seen = set()
                for score, position in zip(scores[0], positions[0]):
                    if position < 0 or position >= len(snapshot.records):
                        continue
                    chunk = snapshot.records[position]
                    key = (chunk.document_id, chunk.pdf_page, chunk.text)
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append(Source(source_id=f"S{len(sources) + 1}", chunk_id=chunk.id,
                        document_id=chunk.document_id, filename=chunk.filename, pdf_page=chunk.pdf_page,
                        passage=chunk.text, score=float(score), extraction_method=chunk.extraction_method))
            return SearchResult(sources=sources, collection_version=self.storage.collection_version(),
                embedding_model=self.engine.model_id, latency_ms=round((perf_counter() - started) * 1000, 2))
