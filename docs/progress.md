# DocuAssist build progress

The local RAG pipeline is implemented and verified on Windows with Python 3.12 and a 16 GB RAM laptop. Version 0.3.0 supports PDF ingestion/OCR, page-aware token chunking, persistent MiniLM embeddings, FAISS retrieval, real local-model generation, citations, and source inspection.

**Verified behavior**

- All 62 automated checks pass, including the original 34 ingestion/OCR checks and new retrieval/citation/lifecycle checks. Controlled model outputs test plumbing; a real MiniLM integration check verifies normalized 384-dimensional vectors and relevant ranking.
- A real text-PDF question produced an answer using the VPN prerequisites on PDF page 2. A separate real scanned-PDF run completed OCR -> embeddings -> retrieval -> Qwen generation, with the citation labelled OCR and PDF page 2.
- Streamlit AppTest submitted a real question through the running HTTP backend, displayed source pages, and cleared its old answer after a collection mutation, with no Streamlit exceptions. This is an app-harness check, not a browser screenshot test.
- Existing extracted documents were indexed at startup without changing their original identities or PDFs. A private database backup was saved before migration.
- Duplicate uploads reuse vectors. Restart reuses matching persisted vectors and rebuilds FAISS. Deletion removes vectors and never returns the deleted passages after restart.
- Re-extraction refreshes page text and vectors atomically. Vector/database failures preserve previous pages and search state. Concurrent answering/deletion uses one collection lock.
- Unknown citation IDs, including IDs hidden in grouped markers, are rejected. Missing inline markers are rendered from valid structured references. Refusals/clarification omit answer-support citations. Provider timeouts remain provider errors.

The local app runs at http://127.0.0.1:8511, the API at http://127.0.0.1:8011, and the default model server at http://127.0.0.1:8012. The default generator is Qwen2.5-1.5B-Instruct Q4_K_M with llama.cpp b11574 on CPU. MiniLM uses a pinned revision and ONNX Runtime. OCR uses PDFium/RapidOCR with packaged CPU models.

**Measured synthetic evaluation**

The current development run has 20 questions, 14 answerable. A fresh holdout_v2 run has 10 questions, 7 answerable. Both used the five fictional manuals, independent questions, 200-token chunks, 30-token overlap, top five retrieval, and a development-tuned 0.20 evidence gate. Both reports record the same system-prompt SHA-256. The fresh holdout was authored after the final prompt change and was not used to tune the released implementation; its question groups are disjoint from the original development/final groups.

| Measure | Current development | Fresh holdout_v2 |
|---|---|---|
| Supporting page hit@5, vector retrieval | 14/14 | 7/7 |
| Supporting page hit@5, TF-IDF baseline | 14/14 | 7/7 |
| All expected supporting pages present@5 | 14/14 | 7/7 |
| Expected answer-status agreement | 20/20 | 9/10 |
| HTTP/provider errors | 0 | 0 |
| Median end-to-end latency | 11.03 s | 10.42 s |
| p95 end-to-end latency | 21.07 s | 14.84 s |

These are source-presence and status measures, not factual answer accuracy. The corpus has only ten pages, so retrieving five pages is a generous setting. TF-IDF also found all expected pages; this run does not show that semantic retrieval outperforms the baseline. Wall-clock timings came from local development runs, with some shared model/CPU activity; they are not isolated performance benchmarks.

Manual review found useful core answers and real source mappings, but also material model-quality limitations. The current development instruction-weekend response begins with No but wrongly treats the quoted instruction as a priority policy. Several responses cite unnecessary passages, exceed the requested length, or omit expected detail. In holdout_v2, the vague unspecified-problem question receives insufficient_evidence instead of a clarification question; the seven answerable responses contain their expected core facts. Valid citation IDs do not establish that every generated claim is supported.

Actual questions, expected answers, generated text, passages, and stage timings for the released prompt are recorded in evaluation/results/dev.json and evaluation/results/holdout_v2.json. Evaluation collections contain only fictional manuals and are isolated from the user's runtime uploads. Model-quality improvements should use development data; using holdout examples to tune future changes requires another fresh held-out set.

The earlier evaluation/results/final.json is retained as a historical run of a retired prompt. A development response copied a factual example from that prompt into an unrelated missing-information answer. All factual examples were removed before the current runs. The current replacement question correctly returns insufficient_evidence. The historical final report does not measure the released prompt and must not be combined with its current results.

**Architecture status**

| Step | Status |
|---|---|
| 1-3: requirements, environment, samples | Implemented |
| 4-5: storage and ingestion/OCR | Implemented; schema version 3 |
| 6-7: token chunks and embeddings | Implemented with limits, fingerprints and persisted vectors |
| 8-9: FAISS and retrieval/baseline | Implemented |
| 10: deletion/recovery | Implemented for originals, pages and search state |
| 11-12: generation/answering | Real llama.cpp/Qwen path verified; optional Ollama adapter implemented but not real-model evaluated |
| 13-14: API/interface | Search/ask endpoints, question form and source display implemented |
| 15: checks/evaluation | Automated lifecycle checks and real-model reports implemented; model-quality limitations recorded |
| 16: packaging/demo | Local setup/launch scripts and demo guide implemented; Docker/production deployment remains an extension |

This release is single-owner and local. Conversation memory, multi-user authentication/collection isolation, richer layout/table handling, and larger-scale deployment are not implemented. Printed upright English OCR is the verified scan scope. Preserve source review in the workflow when using the small local generator.
