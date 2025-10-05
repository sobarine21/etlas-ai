from __future__ import annotations
import os
import io
import json
from typing import List, Dict, Any
import streamlit as st
import requests

# Optional parsers
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# --- PAGE CONFIG
st.set_page_config(
    page_title="⚡ Autorag AI Search Chatbot",
    page_icon="⚡",
    layout="wide",
)

# --- SECRETS / CONFIG
CF_EMAIL = st.secrets.get("CLOUDFLARE_EMAIL", "")
CF_API_KEY = st.secrets.get("CLOUDFLARE_API_KEY", "")
CF_ACCOUNT_ID = st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_AUTORAG_ID = st.secrets.get("CLOUDFLARE_AUTORAG_ID", "")

# --- SESSION STATE
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "files_ctx" not in st.session_state:
    st.session_state.files_ctx = ""
if "raw_files" not in st.session_state:
    st.session_state.raw_files = []
if "last_autorag" not in st.session_state:
    st.session_state.last_autorag = None


# --- HELPERS
def tag(label: str):
    st.markdown(
        f"<span style='padding:2px 8px;border-radius:999px;background:#eee;font-size:12px'>{label}</span>",
        unsafe_allow_html=True,
    )


def extract_text_from_upload(file) -> str:
    name = file.name.lower()
    data = file.read()
    try:
        file.seek(0)
    except Exception:
        pass

    if name.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode(errors="ignore")

    if name.endswith(".pdf") and PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return "\n\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as e:
            st.warning(f"PDF parse failed: {e}")
            return ""

    if name.endswith((".docx", ".doc")) and docx is not None:
        try:
            docf = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in docf.paragraphs])
        except Exception as e:
            st.warning(f"DOCX parse failed: {e}")
            return ""

    st.info(f"Unsupported file type: {file.name}")
    return ""


def call_autorag_search(query: str) -> Dict[str, Any]:
    """Calls Cloudflare Autorag AI Search per official API spec"""
    if not (CF_EMAIL and CF_API_KEY and CF_ACCOUNT_ID and CF_AUTORAG_ID):
        return {"error": "Missing Cloudflare credentials"}

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/autorag/rags/{CF_AUTORAG_ID}/ai-search"

    headers = {
        "Content-Type": "application/json",
        "X-Auth-Email": CF_EMAIL,
        "X-Auth-Key": CF_API_KEY,
    }
    payload = {"query": query}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if not resp.ok:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def build_query(user_input: str, use_files: bool) -> str:
    """Combine user input with uploaded file context"""
    if use_files and st.session_state.files_ctx:
        snippet = st.session_state.files_ctx[:8000]
        return f"{user_input}\n\n=== Uploaded File Context ===\n{snippet}"
    return user_input


# --- SIDEBAR CONFIG
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Configure context and manage session")

    use_files = st.toggle("Use uploaded files as context", value=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 New Chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_autorag = None
            st.toast("Chat cleared")
    with colB:
        if st.button("⬇️ Export Chat", use_container_width=True):
            payload = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            st.download_button(
                "Download JSON",
                data=payload,
                file_name="autorag_chat.json",
                mime="application/json",
            )


# --- MAIN LAYOUT
st.title("⚡ Cloudflare Autorag AI Search Chatbot")
st.markdown("Chat directly with Cloudflare’s native AI Search engine — no LLM key needed.")

# Upload files
uploads = st.file_uploader(
    "Upload files (PDF, TXT, MD, DOCX)",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
)

if uploads:
    st.session_state.raw_files = uploads
    texts = []
    with st.status("Processing uploads…"):
        for f in uploads:
            t = extract_text_from_upload(f)
            if t:
                texts.append(f"# {f.name}\n\n{t}")
    st.session_state.files_ctx = "\n\n\n".join(texts)
    st.success(f"Processed {len(texts)} file(s).")

if st.session_state.files_ctx:
    with st.expander("Preview extracted file text (first 2,000 chars)"):
        st.code(st.session_state.files_ctx[:2000] + ("…" if len(st.session_state.files_ctx) > 2000 else ""))

# Render chat history
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- CHAT INPUT
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        query = build_query(prompt, use_files)
        data = call_autorag_search(query)

        if data.get("error"):
            full_text = f"❌ Error: {data['error']}"
        else:
            result = data.get("result") or data
            if isinstance(result, dict):
                hits = result.get("hits") or result.get("results") or []
            else:
                hits = []

            if not hits:
                full_text = "⚠️ No relevant results found."
            else:
                texts = []
                for i, h in enumerate(hits, start=1):
                    txt = h.get("text") or h.get("chunk") or h.get("content") or ""
                    src = h.get("source") or (h.get("metadata") or {}).get("source") or h.get("url") or ""
                    entry = f"**Result {i}**\n\n{txt.strip()}"
                    if src:
                        entry += f"\n\n*Source:* {src}"
                    texts.append(entry)
                full_text = "\n\n---\n\n".join(texts)

        placeholder.markdown(full_text)
        st.session_state.history.append({"role": "assistant", "content": full_text})
        st.session_state.last_autorag = data


# --- FOOTER
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    tag("Tip")
    st.caption("Upload files to enrich your Autorag search context.")
with col2:
    tag("Privacy")
    st.caption("Files are processed only in-memory during your session.")
with col3:
    tag("Engine")
    st.caption("Powered by Cloudflare Autorag AI Search API.")
