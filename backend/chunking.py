"""Overlapping token windows confined to their original PDF page."""
import hashlib

from backend.schemas import ChunkRecord, Document, Page


def source_digest(pages: list[Page]) -> str:
    digest = hashlib.sha256()
    for page in pages:
        digest.update(f"{page.pdf_page}:{page.extraction_method}:".encode())
        digest.update(page.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def split_page(text, tokenizer, size=200, overlap=30):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoded.offsets
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        # Prefer a nearby sentence/paragraph ending, without starving the next window.
        if end < len(offsets):
            for boundary in range(end, max(start + overlap + 1, end - 25), -1):
                char_end = offsets[boundary - 1][1]
                gap = text[char_end:offsets[boundary][0]]
                if text[char_end - 1:char_end] in ".!?" or "\n" in gap:
                    end = boundary
                    break
        passage = text[offsets[start][0]:offsets[end - 1][1]].strip()
        count = len(tokenizer.encode(passage).ids)
        while count > size + 2 and end > start + 1:
            end -= 1
            passage = text[offsets[start][0]:offsets[end - 1][1]].strip()
            count = len(tokenizer.encode(passage).ids)
        if passage:
            if count > 256:
                raise ValueError("Chunk exceeds the embedding sequence limit")
            yield passage, count
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)


def create_chunks(document: Document, pages: list[Page], engine, settings) -> list[ChunkRecord]:
    records = []
    for page in pages:
        for ordinal, (text, count) in enumerate(split_page(
            page.text, engine.tokenizer, settings.chunk_tokens, settings.chunk_overlap
        )):
            identifier = hashlib.sha256(
                f"{document.id}:{page.pdf_page}:{ordinal}:{engine.fingerprint}:{text}".encode()
            ).hexdigest()
            records.append(ChunkRecord(identifier, document.id, document.filename, page.pdf_page,
                ordinal, text, count, b"", engine.dimension, engine.fingerprint, page.extraction_method))
    vectors = engine.encode([record.text for record in records])
    from dataclasses import replace
    return [replace(record, vector=vector.astype("<f4").tobytes())
            for record, vector in zip(records, vectors)]
