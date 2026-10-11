# DocuAssist

A document question-answering portfolio project built step by step. The first working milestone uploads text or scanned PDFs, extracts text page by page, stores it in SQLite, and lets you inspect the source pages in Streamlit. Local OCR reads image-only pages automatically. Semantic search and generated answers are the next milestones.

Built with Python, FastAPI, Streamlit, SQLite, pypdf, PDFium, and RapidOCR/ONNX Runtime.

![Local OCR reading a fictional VPN manual](docs/ocr-preview.jpg)

**Run locally**

Use Python 3.11 or 3.12. From the project root, create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The project was tested with Python 3.12. Install the pinned dependencies in an isolated virtual environment.

Start the backend and interface in separate terminals:

```powershell
.\scripts\run-backend.ps1
```

```powershell
.\scripts\run-frontend.ps1
```

Open the [interface](http://127.0.0.1:8511) and the [interactive API documentation](http://127.0.0.1:8011/docs). Both services bind to localhost. Use a single backend worker.

Upload a PDF from the sample_docs directory. The fictional VPN guide has prerequisites on PDF page 2. Inspect the extracted text, re-upload the same PDF to check duplicate handling, and remove it using the document controls.

For OCR, upload the image-only fictional-scanned-vpn-guide.pdf. Both pages should show the OCR label, with the approved client and MFA prerequisites retained on PDF page 2. Text PDFs use direct extraction; pages without embedded text are rendered with PDFium and read by RapidOCR on CPU. The recognition models ship with the Python package. After installing dependencies, document processing uses no cloud OCR service, API key, or separate Tesseract installation.

OCR loads on the first scanned upload. Defaults are 180 DPI, at most 4 million rendered pixels per page, and 10 pages needing OCR per PDF. PDFs exceeding that OCR page count are rejected before recognition; split them into smaller files. These settings and the OCR enable switch are in .env.example. /health reports whether OCR is disabled, unavailable, not yet loaded, ready, or failed.

**Current API**

| Method | Endpoint | Behavior |
|---|---|---|
| GET | /health | Capabilities, OCR engine state, collection size, and upload limits |
| POST | /documents | PDF upload; 201 for a new document and 200 for a duplicate |
| GET | /documents | Persistent document list |
| GET | /documents/{id}/pages | Text, extraction_method (text/ocr/none), and warnings in PDF page order |
| POST | /documents/{id}/reextract | Re-extract the stored original with OCR, retaining document identity |
| DELETE | /documents/{id} | Remove document records and original file |

SQLite records and uploaded originals live in the ignored data directory. Set configuration through environment variables or a local .env based on .env.example. Filenames supplied by users are display labels; generated document IDs determine storage locations. No language model or API key is required for this milestone.

**Verify and regenerate samples**

```powershell
.\scripts\run-tests.ps1
.\.venv\Scripts\python.exe scripts\create_sample_documents.py
.\.venv\Scripts\python.exe scripts\create_scanned_sample.py
```

The 34 automated checks cover valid page extraction, duplicate handling, invalid/password-protected/blank PDFs, partial extraction warnings, upload limits, filename traversal, persistence, deletion, and recovery from interrupted cleanup. OCR checks include real recognition of an image-only PDF, blank-page recognition, mixed page provenance, engine failure, rendering bounds, and migration of existing SQLite records. Re-extraction checks cover real OCR on an older mixed PDF, persistent document identity, rollback after a mid-transaction database failure, missing/changed originals, disabled OCR, and protection of previously readable pages. The five sample manuals are fictional and their source text is checked in for reproducibility; the OCR fixture is a rasterized copy of the VPN manual.

Existing databases migrate automatically to schema version 2 while retaining page text, warnings, document IDs, and original PDFs. Existing extracted documents keep their stored text until reprocessed. Select an older PDF and click Re-extract with OCR to process its stored original; no deletion or upload is needed. The original file, document ID, filename, hash and creation timestamp stay the same. Updated page records and counts commit in one database transaction. Fatal extraction or write failures retain the earlier records; an attempt that loses text on a previously readable page is rejected. An exact duplicate upload reuses the latest stored result.

The evaluation directory contains development and held-out question sets for later retrieval and answer evaluation. There are no AI accuracy claims yet. See the [architecture plan](docs/architecture-plan.md) and [build progress](docs/progress.md) for the full sequence and completed work.

**Known boundaries**

The OCR path is verified on clean, upright printed English scans. Rotate sideways or upside-down scans before uploading. The line-angle classifier is disabled because it incorrectly flipped an upright prerequisite line in the real fixture. OCR can misread words, omit lines, and lose spacing; its label is provenance, not an accuracy guarantee. Review the extracted text against the scan. Handwriting, complex tables, and arbitrary page layouts have not been validated. Pages with existing embedded text use that text, so image content on the same page is not additionally recognized.

The project does not yet preserve table structure, authenticate multiple users, search semantically, or generate answers. Limits are intended for a small local demonstration. Public multi-user hosting requires the later authentication and collection-isolation design.
