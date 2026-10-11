import os

import httpx
import streamlit as st

API_URL = os.environ.get("DOCUASSIST_API_URL", "http://127.0.0.1:8011").rstrip("/")

st.set_page_config(page_title="DocuAssist | Document workspace", page_icon="📄", layout="wide")
st.title("DocuAssist")
st.write("Upload a text or scanned PDF and inspect the text extracted from each page.")
st.caption("Build milestone 1 · Document ingestion. Search and generated answers are next.")


def api_request(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, API_URL + path, timeout=300, trust_env=False, **kwargs)
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
    if health["capabilities"].get("ocr"):
        st.caption(f"OCR is automatic for image-only pages · "
                   f"up to {limits['max_ocr_pages']} OCR pages per PDF")
    else:
        st.warning("OCR is unavailable or disabled. Text PDFs can still be uploaded.")
    with st.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader("Choose a PDF (text or scanned)", type=["pdf"])
        submitted = st.form_submit_button("Upload and extract", type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Choose a PDF first.")
        else:
            with st.spinner("Extracting pages and running OCR where needed…"):
                result = api_request(
                    "POST", "/documents",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if result:
                verb = "Already in the collection" if result["duplicate"] else "Extracted"
                st.session_state["notice"] = f"{verb}: {result['document']['filename']}"
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
