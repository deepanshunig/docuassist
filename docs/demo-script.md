# DocuAssist RAG demo

Start the model, backend, and frontend using the README commands. Confirm /health reports semantic_search and generated_answers true.

1. Upload fictional-vpn-guide.pdf. Confirm it is indexed, with two extracted PDF pages.
2. Select the manual in Search in and ask: "What must I do before connecting to the VPN?" Show the answer, expand its source, and point out the client/MFA prerequisites on PDF page 2.
3. Switch to Find passages and repeat the question. Explain that the embedding/FAISS stage retrieves evidence separately from the generator.
4. Ask: "What is my salary payment date?" Show the insufficient-evidence response. Similarity alone does not prove the requested information is present.
5. Ask: "It failed. Which steps should I follow?" Show the request to identify the service or device.
6. Inspect extracted pages. Upload the scanned VPN fixture and repeat the supported question, showing the OCR source label and PDF page number.
7. Re-upload identical bytes to demonstrate reuse. Delete a demo document, restart the backend, and confirm its passages stay absent. Delete only the demonstration upload you intend to remove.

Interview points: explain token overlap and page provenance; distinguish embedding vectors from generated text; explain SQLite as canonical storage and FAISS as derived state; demonstrate failure/rollback handling; distinguish citation-ID validity from factual support; show evaluation failures as well as successes.

Limits to state: single local owner, independent questions, a small CPU model, printed English OCR, and synthetic evaluation. Authentication, conversation memory, richer layouts, and container/production deployment remain extensions.
