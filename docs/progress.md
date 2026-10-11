**DocuAssist build progress**

Milestone 1 is verified: requirements, the Python environment, five fictional sample manuals,
development/final evaluation records, SQLite storage, PDF upload/extraction, and a working
document inspection interface, now with local OCR for image-only pages. All 34 ingestion/OCR checks pass. The 30 evaluation questions have
valid source pages and isolated question groups. All 10 sample PDF pages were rendered and
visually inspected. A browser upload of the VPN guide displayed its two extracted pages correctly.

The interface runs at http://127.0.0.1:8511 and the backend at http://127.0.0.1:8011.
The verified interface screenshot is saved as milestone-1.jpg beside this document.

OCR uses PDFium rendering and RapidOCR with packaged ONNX models on CPU. The real two-page
image-only VPN fixture recovers the client and MFA prerequisites on PDF page 2. The test suite
also covers OCR bypass for text PDFs, blank scans, mixed page numbering, missing/broken OCR,
page limits, bounded raster sizes, OCR persistence/deletion, and version 1 database migration.
Defaults are 180 DPI, 4 million pixels per rendered page, and at most 10 pages needing OCR
per PDF. Upright printed English scans are the verified scope; OCR text still needs review.
The line-angle classifier is disabled after it incorrectly flipped one upright fixture line.
The running backend is version 0.2.1. A browser upload of the scanned fixture displayed two
OCR pages, including the client/MFA prerequisites on page 2; the preview is ocr-preview.jpg.
The existing collection remained intact after migration. Previously extracted documents keep
their stored results until Re-extract with OCR is selected. This processes the stored PDF,
preserves its identity and original, and commits refreshed pages/counts atomically. Tests verify
real OCR on older mixed PDFs and preservation of prior results after failed re-extraction.
A browser retry of an older mixed PDF replaced its legacy missing-text warning with an OCR
page label and recovered page text. Both pages were readable afterward, with the original
PDF hash and collection count unchanged. Personal-document previews remain in ignored cache.

The verified environment uses Python 3.12.14. Create an isolated virtual environment and
install the pinned dependency files to run the project on another machine.

| Architecture step | Implementation status |
|---|---|
| 1. Requirements | Written with acceptance scenarios |
| 2. Environment | Project virtual environment and launcher scripts created |
| 3. Demo/evaluation data | Five manuals and 20 development / 10 final questions prepared |
| 4. Storage/contracts | Documents, pages, cleanup queue, API schemas implemented; chunk/vector storage follows in steps 6–7 |
| 5. Ingestion | Upload validation, direct extraction, automatic local OCR, page provenance, warnings and exact-duplicate handling implemented |
| 6–9. Chunking/search | Next milestone |
| 10. Deletion/recovery | Initial document deletion and recovery implemented; search lifecycle follows with indexing |
| 11–12. Generation | Planned |
| 13. API | Document endpoints implemented; search/ask follow later |
| 14. Interface | Upload, inspection and deletion implemented; answering interface follows later |
| 15. Evaluation/checks | 34 ingestion/OCR checks and evaluation records prepared; retrieval/answer metrics follow later |
| 16. Packaging | Planned |

The next implementation target is tokenizer-aware chunking using the actual embedding model's
tokenizer, then persistent embeddings and FAISS search. Current document statuses deliberately
say extracted rather than indexed. There is no generated-answer feature or measured AI accuracy yet.
