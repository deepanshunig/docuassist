import os

import httpx
import streamlit as st

API_URL = os.environ.get("DOCUASSIST_API_URL", "http://127.0.0.1:8011").rstrip("/")

st.set_page_config(page_title="DocuAssist | Document workspace", page_icon="📄", layout="wide")
st.title("DocuAssist")
st.write("Ask questions about your PDFs and inspect the passages used to answer them.")
st.caption("Local RAG document assistant · Text PDFs and scanned pages with OCR")


def api_request(method: str, path: str, **kwargs):
    try:
        timeout = kwargs.pop("timeout", 300)
        response = httpx.request(method, API_URL + path, timeout=timeout, trust_env=False, **kwargs)
        data = response.json()
        if response.is_error:
            error = data.get("error", {})
            message = error.get("message") if isinstance(error, dict) else None
            st.error(message or f"Request failed ({response.status_code}).")
            return None
        return data
    except (httpx.HTTPError, ValueError):
        st.error("Cannot reach the backend. Start the backend and refresh this page.")
        return None


health = api_request("GET", "/health")
if health is None:
    st.stop()

version = health["rag"]["retrieval"]["collection_version"]
if st.session_state.get("collection_version") != version:
    st.session_state.pop("question_result", None)
    st.session_state["collection_version"] = version


def show_sources(sources, expanded=False):
    for source in sources:
        with st.expander(f"[{source['source_id']}] {source['filename']} · PDF page {source['pdf_page']}",
                         expanded=expanded):
            st.text(source["passage"])
            st.caption(f"Retrieval score: {source['score']:.3f} · "
                       f"{'OCR text' if source['extraction_method'] == 'ocr' else 'PDF text'}")

if "notice" in st.session_state:
    st.success(st.session_state.pop("notice"))
if "cleanup_warnings" in st.session_state:
    for warning in st.session_state.pop("cleanup_warnings"):
        st.warning(warning)

documents = api_request("GET", "/documents")
if documents is None:
    st.stop()

limits = health["limits"]
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Your documents")
    st.caption(f"{len(documents)} / {limits['max_documents']} documents · "
               f"{limits['max_upload_mb']} MiB and {limits['max_pages']} pages per PDF")
    if health["capabilities"]["semantic_search"]:
        retrieval = health["rag"]["retrieval"]
        st.caption(f"{retrieval['indexed_document_count']} searchable documents · {retrieval['chunk_count']} passages")
    else:
        st.warning("Search needs its local embedding model. Run scripts/setup_models.py and restart the backend.")
    if health["capabilities"].get("ocr"):
        st.caption(f"OCR is automatic for image-only pages · "
                   f"up to {limits['max_ocr_pages']} OCR pages per PDF")
    else:
        st.warning("OCR is unavailable or disabled. Text PDFs can still be uploaded.")
    with st.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader("Choose a PDF (text or scanned)", type=["pdf"])
        submitted = st.form_submit_button("Upload PDF", type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Choose a PDF first.")
        else:
            with st.spinner("Extracting pages, running OCR where needed, and indexing passages…"):
                result = api_request(
                    "POST", "/documents",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if result:
                verb = "Already in the collection" if result["duplicate"] else (
                    "Indexed" if result["indexed"] else "Extracted")
                st.session_state["notice"] = f"{verb}: {result['document']['filename']}"
                if result.get("indexing_warning"):
                    st.session_state["cleanup_warnings"] = [result["indexing_warning"]]
                st.session_state["selected_id"] = result["document"]["id"]
                st.rerun()

    if not documents:
        st.info("Start by uploading a sample manual from the project's sample_docs folder.")
    else:
        ids = [document["id"] for document in documents]
        document_map = {document["id"]: document for document in documents}
        if st.session_state.get("selected_id") not in ids:
            st.session_state["selected_id"] = ids[0]
        selected_id = st.selectbox(
            "Inspect document", ids, key="selected_id",
            format_func=lambda identifier: document_map[identifier]["filename"],
        )
        with st.expander("Remove document"):
            confirmed = st.checkbox("Remove this PDF and its extracted pages", key=selected_id)
            if st.button("Delete document", disabled=not confirmed):
                result = api_request("DELETE", f"/documents/{selected_id}")
                if result:
                    st.session_state["notice"] = "Document removed."
                    st.session_state["cleanup_warnings"] = result["warnings"]
                    st.session_state.pop("selected_id", None)
                    st.rerun()

with right:
    question_tab, pages_tab = st.tabs(["Ask your documents", "Inspect extracted pages"])
    with question_tab:
        st.subheader("Ask your documents")
        mode = st.radio("Choose an action", ["Get an answer", "Find passages"], horizontal=True)
        scope_ids = [None] + [document["id"] for document in documents]
        names = {document["id"]: document["filename"] for document in documents}
        scope = st.selectbox("Search in", scope_ids,
            format_func=lambda identifier: "All documents" if identifier is None else names[identifier])
        can_search = bool(documents) and health["capabilities"]["semantic_search"]
        can_answer = can_search and health["capabilities"]["generated_answers"]
        if not documents:
            st.info("Upload a PDF, then ask a question about its contents.")
        elif mode == "Get an answer" and not health["capabilities"]["generated_answers"]:
            st.info("The local answer model is offline. Start scripts/run-model.ps1. Passage search is available when indexing is ready.")
        with st.form("question_form"):
            question = st.text_input("Your question", max_chars=2000,
                placeholder="What must I do before connecting to the VPN?")
            submitted_question = st.form_submit_button(
                "Ask documents" if mode == "Get an answer" else "Find passages",
                type="primary", disabled=not (can_answer if mode == "Get an answer" else can_search))
        if submitted_question:
            st.session_state.pop("question_result", None)
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                payload = {"question": question, "document_ids": None if scope is None else [scope]}
                endpoint = "/ask" if mode == "Get an answer" else "/search"
                with st.spinner("Retrieving passages and preparing an answer..." if endpoint == "/ask" else "Finding relevant passages..."):
                    result = api_request("POST", endpoint, json=payload,
                        timeout=health["rag"]["answer_timeout_seconds"])
                if result:
                    st.session_state["question_result"] = {"kind": endpoint, "data": result}
        saved = st.session_state.get("question_result")
        if saved:
            result = saved["data"]
            if saved["kind"] == "/ask":
                if result["status"] == "insufficient_evidence":
                    st.info(result["answer"])
                elif result["status"] in {"ambiguous", "clarification"}:
                    st.warning(result["answer"])
                else:
                    st.markdown(result["answer"])
                st.caption(f"{result['latency_ms'] / 1000:.1f} seconds" +
                           (f" · {result['model']}" if result["model"] else ""))
                if result["sources"]:
                    st.subheader("Supporting passages")
                    show_sources(result["sources"], expanded=True)
                cited = {s["source_id"] for s in result["sources"]}
                other = [s for s in result["retrieved_sources"] if s["source_id"] not in cited]
                if other:
                    with st.expander("Other retrieved passages"):
                        show_sources(other)
            else:
                if result["sources"]:
                    show_sources(result["sources"], expanded=True)
                    st.caption("Retrieval scores measure similarity, not answer correctness.")
                else:
                    st.info("No searchable passages in this selection.")
        st.caption("Check answers against the supporting passages. Page numbers use the PDF file order.")
    with pages_tab:
        st.subheader("Extracted pages")
        if not documents:
            st.write("Uploaded text will appear here with its PDF page number.")
            st.write("Try the fictional VPN guide first. Its prerequisites appear on PDF page 2.")
        else:
            result = api_request("GET", f"/documents/{selected_id}/pages")
            if result:
                document = result["document"]
                missing_count = sum(not page["text"] for page in result["pages"])
                if missing_count and health["capabilities"].get("ocr"):
                    st.info("Some pages have no saved text. Re-extract this PDF to try OCR on them.")
                if st.button("Re-extract with OCR", disabled=not health["capabilities"].get("ocr")):
                    with st.spinner("Re-extracting the stored PDF with OCR…"):
                        refreshed = api_request("POST", f"/documents/{selected_id}/reextract")
                    if refreshed:
                        read_count = refreshed["document"]["text_page_count"]
                        total_count = refreshed["document"]["page_count"]
                        st.session_state["notice"] = (
                            f"Re-extracted: {refreshed['document']['filename']} · "
                            f"{read_count} / {total_count} pages with text"
                        )
                        st.rerun()
                ocr_count = sum(page["extraction_method"] == "ocr" for page in result["pages"])
                first, second, third, fourth = st.columns(4)
                first.metric("PDF pages", document["page_count"])
                second.metric("Pages with text", document["text_page_count"])
                third.metric("OCR pages", ocr_count)
                fourth.metric("Size", f"{document['size_bytes'] / 1024:.1f} KiB")
                st.caption(document["filename"] + " · Page numbers refer to the PDF file order.")
                if ocr_count:
                    st.info("OCR can misread characters. Check the extracted text against the scan.")
                for page in result["pages"]:
                    method = {"text": "PDF text", "ocr": "OCR", "none": "No text"}[
                        page["extraction_method"]
                    ]
                    with st.expander(f"PDF page {page['pdf_page']} · "
                                     f"{method} · {page['character_count']} characters", expanded=True):
                        if page["warning"]:
                            st.warning(page["warning"])
                        st.text(page["text"] or "No extractable text on this page.")
