"""Streamlit interface for the resume parser API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so `app.*` is importable when Streamlit
# runs this script directly (Streamlit adds ui/ but not the parent directory).
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.utils.logger import configure_logging

load_dotenv()

# Guard against re-running on every Streamlit re-render.
if "logging_configured" not in st.session_state:
    configure_logging("ui")
    st.session_state["logging_configured"] = True

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def main() -> None:
    """Render the resume parser UI."""
    st.set_page_config(page_title="Resume Parser", layout="wide")
    _apply_styles()

    st.title("Resume Parser")
    st.caption("Upload a PDF or DOCX resume and extract structured information using an LLM.")

    with st.sidebar:
        st.header("Upload & Extract")
        uploaded_file = st.file_uploader("Select resume", type=["pdf", "docx"])
        extract_clicked = st.button("Extract", type="primary", use_container_width=True)

        st.divider()

        st.header("Lookup by ID")
        doc_id_input = st.text_input("Document ID", placeholder="Paste a document ID…")
        fetch_clicked = st.button("Fetch", use_container_width=True)


    if extract_clicked:
        if uploaded_file is None:
            st.error("Choose a PDF or DOCX resume first.")
        else:
            with st.spinner("Extracting resume data…"):
                try:
                    result = _upload_resume(uploaded_file)
                    st.session_state["resume_result"] = result
                    st.success(f"Extracted — Document ID: `{result['document_id']}`")
                except httpx.HTTPStatusError as exc:
                    st.error(_error_detail(exc.response))
                except httpx.HTTPError as exc:
                    st.error(f"Could not reach the API ({API_BASE_URL}): {exc}")

    if fetch_clicked:
        if not doc_id_input.strip():
            st.error("Enter a document ID.")
        else:
            with st.spinner("Fetching…"):
                try:
                    result = _fetch_resume(doc_id_input.strip())
                    st.session_state["resume_result"] = result
                except httpx.HTTPStatusError as exc:
                    st.error(_error_detail(exc.response))
                except httpx.HTTPError as exc:
                    st.error(f"Could not reach the API: {exc}")

    result = st.session_state.get("resume_result")
    if result:
        _render_resume(result)
    else:
        _render_recent()


def _upload_resume(upload: Any) -> dict[str, Any]:
    files = {
        "file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")
    }
    with httpx.Client(timeout=180) as client:
        response = client.post(f"{API_BASE_URL}/api/upload", files=files)
        response.raise_for_status()
        return response.json()


def _fetch_resume(document_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{API_BASE_URL}/api/resume/{document_id}")
        response.raise_for_status()
        return response.json()


def _fetch_recent(limit: int = 10) -> dict[str, Any]:
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{API_BASE_URL}/api/resumes?limit={limit}")
        response.raise_for_status()
        return response.json()


def _render_resume(result: dict[str, Any]) -> None:
    """Render extracted resume data in organized sections."""
    data = result["extracted_data"]
    contact = data.get("contact", {})

    col_meta, col_dl = st.columns([5, 1])
    with col_meta:
        uploaded = result.get("uploaded_at", "")[:10]
        st.caption(
            f"**{result['filename']}**  |  ID: `{result['document_id']}`  |  {uploaded}"
        )
    with col_dl:
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name=f"resume_{result['document_id'][:8]}.json",
            mime="application/json",
            use_container_width=True,
        )


    cols = st.columns(4)
    cols[0].metric("Name", contact.get("name") or "—")
    cols[1].metric("Email", contact.get("email") or "—")
    cols[2].metric("Phone", contact.get("phone") or "—")
    cols[3].metric("Location", contact.get("location") or "—")

    links = contact.get("links", [])
    if links:
        st.markdown("  ".join(f"[{_truncate(link, 40)}]({link})" for link in links))

    _section("Summary", data.get("summary") or "—")

    left, right = st.columns([1.3, 1])

    with left:
        _experience_section(data.get("work_experience", []))
        _education_section(data.get("education", []))
        _projects_section(data.get("projects", []))

    with right:
        _list_section("Technical Skills", data.get("technical_skills", []))
        _list_section("Soft Skills", data.get("soft_skills", []))
        _list_section("Languages", data.get("languages", []))
        _certification_section(data.get("certifications", []))
        _list_section("Awards & Honors", data.get("awards", []))
        _list_section("Extraction Notes", data.get("raw_extraction_notes", []))


def _render_recent() -> None:
    """Show a placeholder and the most recently parsed resumes."""
    st.info("Upload a resume from the sidebar, or look up an existing document by ID.")

    try:
        data = _fetch_recent(limit=10)
    except httpx.HTTPError:
        return

    items = data.get("items", [])
    if not items:
        return

    st.subheader(f"Recent Parses  ·  {data.get('total', 0)} total")
    for item in items:
        label = item.get("candidate_name") or item.get("filename", "Unknown")
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"**{label}**  \n`{item['document_id']}`")
            c2.caption(item.get("uploaded_at", "")[:10])
            if c3.button("Load", key=f"load_{item['document_id']}", use_container_width=True):
                try:
                    result = _fetch_resume(item["document_id"])
                    st.session_state["resume_result"] = result
                    st.rerun()
                except httpx.HTTPError:
                    st.error("Failed to load.")


def _experience_section(items: list[dict[str, Any]]) -> None:
    st.subheader("Work Experience")
    if not items:
        st.write("—")
        return
    for item in items:
        title = " – ".join(v for v in [item.get("role"), item.get("company")] if v)
        with st.container(border=True):
            st.markdown(f"**{title or 'Experience'}**")
            meta = " | ".join(
                v for v in [item.get("duration"), item.get("location")] if v
            )
            if meta:
                st.caption(meta)
            for resp in item.get("responsibilities", []):
                st.write(f"- {resp}")


def _education_section(items: list[dict[str, Any]]) -> None:
    st.subheader("Education")
    if not items:
        st.write("—")
        return
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.get('degree') or 'Degree'}**")
            meta_parts = [item.get("institution"), item.get("year"), item.get("location")]
            if item.get("gpa"):
                meta_parts.append(f"GPA: {item['gpa']}")
            meta = " | ".join(v for v in meta_parts if v)
            if meta:
                st.caption(meta)


def _projects_section(items: list[dict[str, Any]]) -> None:
    st.subheader("Projects")
    if not items:
        st.write("—")
        return
    for item in items:
        with st.container(border=True):
            header = item.get("name") or "Project"
            url = item.get("url")
            st.markdown(f"**{header}**" + (f"  —  [{url}]({url})" if url else ""))
            if item.get("duration"):
                st.caption(item["duration"])
            if item.get("description"):
                st.write(item["description"])
            if item.get("technologies"):
                _inline_pills(item["technologies"])


def _certification_section(items: list[dict[str, Any]]) -> None:
    st.subheader("Certifications")
    if not items:
        st.write("—")
        return
    for item in items:
        label = " | ".join(
            v for v in [item.get("name"), item.get("issuer"), item.get("year")] if v
        )
        st.write(f"- {label}")


def _list_section(title: str, items: list[str]) -> None:
    st.subheader(title)
    if not items:
        st.write("—")
        return
    _inline_pills(items)


def _inline_pills(items: list[str]) -> None:
    st.markdown(
        '<div class="pill-grid">'
        + "".join(f'<span class="pill">{_esc(item)}</span>' for item in items)
        + "</div>",
        unsafe_allow_html=True,
    )


def _section(title: str, body: str) -> None:
    st.subheader(title)
    st.write(body)


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Request failed."
    return str(payload.get("detail") or "Request failed.")


def _esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[:max_len - 1] + "…"


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        div[data-testid="stMetric"] {
            border: 1px solid #E3E7EE;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #FFFFFF;
        }
        .pill-grid { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 1rem; }
        .pill {
            border: 1px solid #D7DDE8;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            background: #F7F9FC;
            color: #222B38;
            font-size: 0.875rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
