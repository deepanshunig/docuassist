**DocuAssist requirements**

An employee uploads fictional IT manuals and asks questions about their contents. The completed project will answer from retrieved evidence and display document names, PDF page numbers, and supporting passages. It demonstrates ingestion, retrieval, grounded generation, evaluation, and deployment for an AI/ML portfolio.

Release scope: one local owner, one collection, English text PDFs and clean upright printed scans, independent questions, 10 documents, 10 MiB per file, and 50 PDF pages per document. Automatic local OCR supports up to 10 pages without embedded text per upload. Rendered pages are bounded to 4 million pixels by default.

**Acceptance scenarios**

1. Upload the fictional VPN guide. Store two pages and retain prerequisites on PDF page 2. Re-uploading identical bytes reuses the document.
2. Ask what is required before VPN access. The finished answering pipeline should cite the approved client and MFA prerequisites on PDF page 2.
3. Ask about annual leave entitlement. The finished pipeline should explain that the uploaded IT manuals do not contain enough evidence.
4. Upload an invalid or password-protected PDF. Show a useful error and create no searchable content.
5. Upload a PDF containing one blank page and one text page. Preserve page order and show an extraction warning.
6. Delete a document and restart. Its records, page text, and eventual search results remain absent. Report and retry original-file cleanup if the operating system temporarily locks the file.
7. Use a question that cannot identify a particular procedure. The finished answering pipeline should ask for clarification.
8. Encounter an instruction inside a document telling the assistant to ignore policy. The finished pipeline should treat that text as evidence, not an instruction governing its behavior.
9. Upload the image-only VPN fixture. Read both pages using OCR, retain one-based PDF page numbers, show OCR provenance and a review notice, and recover the approved client and MFA prerequisites on PDF page 2.
10. Encounter a local OCR engine failure. Keep readable native text in a mixed PDF with per-page warnings; reject an entirely unreadable PDF without creating document records or originals.
11. Restart with a version 1 database. Migrate the page extraction-method field while retaining existing page text and original files.
12. Select an older uploaded PDF and re-extract with OCR. Process the stored original without deletion or another upload; retain document identity and commit new page records/counts together. Keep earlier records if extraction or database writing fails, or a previously readable page loses its text.

Milestone 1 implements upload, validation, direct extraction, OCR fallback, inspection, persistence, duplicate detection, and deletion. Semantic search and generated answers are later milestones. Document statuses say extracted, not indexed, until embeddings and search are implemented.
