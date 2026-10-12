"""Evaluate real retrieval/answers using only the checked-in fictional manuals."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import PROJECT_ROOT, Settings
from backend.main import create_app
from backend.answering import SYSTEM


def keyword_rank(question, records, k=5):
    """A TF-IDF cosine baseline over exactly the same page chunks as FAISS."""
    counts = [Counter(re.findall(r"\b\w+\b", record.text.lower())) for record in records]
    vocabulary = sorted({word for count in counts for word in count})
    matrix = np.array([[count[word] for word in vocabulary] for count in counts], dtype=np.float32)
    idf = np.log((1 + len(records)) / (1 + (matrix > 0).sum(axis=0))) + 1
    matrix = np.log1p(matrix) * idf
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    query = Counter(re.findall(r"\b\w+\b", question.lower()))
    vector = np.log1p(np.array([query[word] for word in vocabulary])) * idf
    norm = np.linalg.norm(vector)
    scores = matrix @ (vector / norm if norm else vector)
    return [records[index] for index in np.argsort(-scores)[:k]]


def coverage(sources, expected):
    found = {(source["filename"], source["pdf_page"]) for source in sources}
    wanted = {(source["document"], source["pdf_page"]) for source in expected}
    return bool(found & wanted), bool(wanted) and wanted <= found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "final", "holdout_v2"], default="dev")
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    settings = Settings(data_dir=PROJECT_ROOT / ".cache" / "evaluation" / args.split,
                        embedding_dir=Settings().model_dir, ocr_enabled=False)
    questions = [json.loads(line) for line in (PROJECT_ROOT / "evaluation" / f"{args.split}.jsonl").read_text().splitlines()]
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_revision": settings.embedding_revision,
        "generator_model": settings.llm_model, "generator_provider": settings.llm_provider,
        "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "chunk_tokens": settings.chunk_tokens, "chunk_overlap": settings.chunk_overlap,
        "retrieval_k": settings.retrieval_k, "retrieval_min_score": settings.retrieval_min_score,
        "scope": "Five fictional manuals; CPU inference; independent questions",
        "quality_note": "Status agreement is not answer correctness. Answers require manual factual/support review.",
        "records": []}
    with TestClient(create_app(settings)) as client:
        for filename in ("fictional-vpn-guide.pdf", "fictional-password-guide.pdf", "fictional-software-guide.pdf",
                         "fictional-device-guide.pdf", "fictional-support-guide.pdf"):
            path = PROJECT_ROOT / "sample_docs" / filename
            response = client.post("/documents", files={"file": (filename, path.read_bytes(), "application/pdf")})
            response.raise_for_status()
        chunks = client.app.state.document_service.storage.list_chunks()
        for question in questions:
            payload = {"question": question["question"]}
            search = client.post("/search", json=payload)
            search.raise_for_status()
            hits = search.json()["sources"]
            baseline = [{"filename": item.filename, "pdf_page": item.pdf_page}
                        for item in keyword_rank(question["question"], chunks)]
            record = dict(question)
            record.update({"retrieved": hits, "keyword_sources": baseline,
                "vector_hit_at_5": coverage(hits, question["support"])[0],
                "all_support_at_5": coverage(hits, question["support"])[1],
                "keyword_hit_at_5": coverage(baseline, question["support"])[0]})
            if not args.search_only:
                answer = client.post("/ask", json=payload)
                record["http_status"] = answer.status_code
                record["result"] = answer.json()
                record["status_matches"] = answer.status_code == 200 and answer.json()["status"] == question["expected_status"]
                print(f"{question['id']}: {answer.status_code} {answer.json().get('status', 'error')}", flush=True)
            report["records"].append(record)
    answerable = [record for record in report["records"] if record["answerable"]]
    latencies = [r["result"]["latency_ms"] for r in report["records"] if r.get("http_status") == 200]
    report["summary"] = {"questions": len(questions), "answerable_questions": len(answerable),
        "vector_hit_at_5": sum(r["vector_hit_at_5"] for r in answerable) / len(answerable),
        "keyword_hit_at_5": sum(r["keyword_hit_at_5"] for r in answerable) / len(answerable),
        "all_support_at_5": sum(r["all_support_at_5"] for r in answerable) / len(answerable),
        "status_matches": sum(r.get("status_matches", False) for r in report["records"]),
        "provider_errors": sum(r.get("http_status", 200) != 200 for r in report["records"]),
        "median_latency_ms": float(np.median(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None}
    output = PROJECT_ROOT / "evaluation" / "results" / f"{args.split}{'-search' if args.search_only else ''}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
