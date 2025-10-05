from __future__ import annotations
import os
import io
import json
import time
import textwrap
from typing import List, Dict, Any, Optional

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

# --- Page config
st.set_page_config(
    page_title="Autorag AI Search Chat",
    page_icon="🤖",
    layout="wide",
)

# --- THEME HELPERS
def tag(label: str):
    st.markdown(
        f"<span style='padding:2px 8px;border-radius:999px;background:#eee;font-size:12px'>{label}</span>",
        unsafe_allow_html=True,
    )

# --- SECRETS / CONFIG
CF_EMAIL = st.secrets.get("CLOUDFLARE_EMAIL", "")
CF_API_KEY = st.secrets.get("CLOUDFLARE_API_KEY", "")
CF_ACCOUNT_ID = st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_AUTORAG_ID = st.secrets.get("CLOUDFLARE_AUTORAG_ID", "")

# --- Session State
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []  # [{"role":"user|assistant", "content":"..."}]
if "files_ctx" not in st.session_state:
    st.session_state.files_ctx = ""  # concatenated extracted text from uploads
if "raw_files" not in st.session_state:
    st.session_state.raw_files = []  # keep references to UploadedFiles
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_autorag" not in st.session_state:
    st.session_state.last_autorag = None

# --- Utilities

def extract_text_from_upload(file) -> str:
    name = file.name.lower()
    data = file.read()
    # Reset pointer so file can be re-read if needed later
    try:
        file.seek(0)
    except Exception:
        pass

    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode(errors="ignore")

    if name.endswith(".pdf") and PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n\n".join(pages)
        except Exception as e:
            st.warning(f"PDF parsing failed for {file.name}: {e}")
            return ""

    if (name.endswith(".docx") or name.endswith(".doc")) and docx is not None:
        try:
            doc = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.warning(f"DOCX parsing failed for {file.name}: {e}")
            return ""

    st.info(f"Unsupported file type for extraction: {file.name}. Treating as text-less.")
    return ""


def chunk_text(txt: str, max_chars: int = 15000) -> str:
    """Trim huge context. Keep prompts snappy."""
    if len(txt) <= max_chars:
        return txt
    head = txt[: max_chars // 2]
    tail = txt[-max_chars // 2 :]
    return head + "\n\n…\n\n" + tail


def call_cloudflare_autorag_ai_search(query: str, *, max_snippets: int = 5) -> Dict[str, Any]:
    """Query Cloudflare Autorag AI Search endpoint and return JSON response."""
    if not (CF_EMAIL and CF_API_KEY and CF_ACCOUNT_ID and CF_AUTORAG_ID):
        return {"error": "Missing Cloudflare secrets"}

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/"
        f"autorag/rags/{CF_AUTORAG_ID}/ai-search"
    )

    headers = {
        "Content-Type": "application/json",
        "X-Auth-Email": CF_EMAIL,
        "X-Auth-Key": CF_API_KEY,
    }
    payload = {"query": query}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    # Normalize into a simple list of snippets if possible
    snippets = []
    try:
        result = data.get("result") or data
        hits = result.get("hits") or result.get("results") or []
        for h in hits[:max_snippets]:
            txt = h.get("text") or h.get("chunk") or h.get("content") or ""
            src = h.get("source") or (h.get("metadata") or {}).get("source") or h.get("url") or ""
            if txt:
                snippets.append({"text": txt, "source": src, "raw_hit": h})
    except Exception:
        pass

    return {"raw": data, "snippets": snippets}


def generate_response_with_autorag_context(user_query: str, *, use_files: bool, use_autorag: bool) -> str:
    """
    Generates a response using Autorag AI Search and optionally uploaded file context.
    For simplicity, this function directly formats the RAG results into a response.
    In a more advanced setup, you'd integrate an LLM here to synthesize the answer.
    """
    response_parts: List[str] = []

    # Add system instruction content (explicit rules + citation instruction)
    system_instructions = textwrap.dedent(
        """
        You are a helpful AI assistant inside a Streamlit app. Follow the rules:
        - Use the provided context below (if any) to answer the user's question.
        - When using retrieved passages from the Cloudflare Autorag retrieval, explicitly cite them inline as [RAG #] where # matches the snippet numbering.
        - If the answer relies on uploaded files, cite them as [FILE: filename] inline.
        - If multiple sources conflict, state the conflict and provide which source you relied on.
        - Prefer factual, step-by-step answers. If uncertain, say so and provide suggestions to verify.
        - Keep code blocks minimal and runnable where possible.
        """
    ).strip()
    response_parts.append(f"**Instructions:**\n{system_instructions}\n")

    # Add uploaded files context (if requested)
    if use_files and st.session_state.files_ctx:
        files_excerpt = chunk_text(st.session_state.files_ctx, 12000)
        response_parts.append(f"**=== Uploaded files context ===**\n\n{files_excerpt}\n")

    # Add Autorag snippets if requested
    if use_autorag:
        rag = call_cloudflare_autorag_ai_search(user_query)
        st.session_state.last_autorag = rag
        if rag.get("error"):
            response_parts.append(f"**[Autorag error]** {rag['error']}\n")
        else:
            snips = rag.get("snippets", [])
            if snips:
                formatted_snips = []
                for i, s in enumerate(snips, start=1):
                    src = f" (source: {s['source']})" if s.get("source") else ""
                    snippet_text = s['text']
                    snippet_text = snippet_text if len(snippet_text) <= 4000 else snippet_text[:4000] + "\n\n[TRUNCATED]"
                    formatted_snips.append(f"[RAG {i}]{src}\n{snippet_text}")
                response_parts.append(f"**=== Cloudflare Autorag retrieval (numbered) ===**\n\n" + "\n\n---\n\n".join(formatted_snips) + "\n")
            else:
                response_parts.append("**Cloudflare Autorag returned no snippets.**\n")

    response_parts.append(f"**User Query:** {user_query}\n")

    # A simple placeholder for where an LLM would synthesize an answer based on the context.
    # For now, we'll just present the context and indicate an LLM would normally answer.
    if use_autorag or use_files:
        response_parts.append("\n**A large language model would now synthesize an answer based on the above context.**")
    else:
        response_parts.append("\n**No context provided. A large language model would normally answer your query directly.**")

    return "".join(response_parts)


# --- SIDEBAR CONFIG
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Configure retrieval + context")

    # Removed model and temperature sliders as Gemini is no longer used
    use_autorag = st.toggle("Use Cloudflare Autorag AI Search", value=True)
    use_files = st.toggle("Use uploaded files as context", value=True)

    with st.expander("Cloudflare Autorag credentials (read-only)"):
        st.text_input("Account ID", value="**HIDDEN**", disabled=True)
        st.text_input("RAG ID", value="**HIDDEN**", disabled=True)
        st.text_input("Email", value="**HIDDEN**", disabled=True)
        st.text_input("API Key", value="**HIDDEN**", disabled=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 New chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_response = ""
            st.toast("Chat cleared")
    with colB:
        if st.button("⬇️ Export chat (JSON)", use_container_width=True):
            payload = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            st.download_button(
                label="Download now",
                data=payload,
                file_name="chat_history.json",
                mime="application/json",
            )

# --- MAIN LAYOUT
st.title("🤖 Autorag AI Search Chatbot")

st.markdown(
    "Ask questions, upload files, and augment with Cloudflare Autorag retrieval."
)

# File uploader
uploads = st.file_uploader(
    "Upload files (PDF, TXT, MD, DOCX)",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
)

if uploads:
    # Store raw refs and build context
    st.session_state.raw_files = uploads
    texts = []
    with st.status("Processing uploads…", expanded=False):
        for f in uploads:
            t = extract_text_from_upload(f)
            if t:
                texts.append(f"# File: {f.name}\n\n" + t)
    st.session_state.files_ctx = "\n\n\n".join(texts)
    st.success(f"Processed {len(texts)} file(s).")

# Show a preview of the file context (collapsed)
if st.session_state.files_ctx:
    with st.expander("Preview extracted file text (first 2,000 chars)"):
        st.code(st.session_state.files_ctx[:2000] + ("…" if len(st.session_state.files_ctx) > 2000 else ""))

# Optionally show last Autorag response for debugging (collapsed)
with st.expander("Debug: last Autorag raw response (collapse)"):
    st.write(st.session_state.last_autorag)

# Render previous chat
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])  # content is already markdown-safe

# Prompt input
prompt = st.chat_input("Type your message…")

# Handle submit
if prompt:
    # Echo user message
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare assistant message container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        # Guardrails: check Cloudflare keys
        if not (CF_EMAIL and CF_API_KEY and CF_ACCOUNT_ID and CF_AUTORAG_ID):
            st.error("Missing Cloudflare Autorag credentials in Streamlit secrets.")
        else:
            try:
                # Generate response based on Autorag and file context
                full_text = generate_response_with_autorag_context(prompt, use_files=use_files, use_autorag=use_autorag)
                placeholder.markdown(full_text)
                st.session_state.last_response = full_text
            except Exception as e:
                st.error(f"Error generating response: {e}")
                full_text += f"\n\n*Error: {e}*"
                placeholder.markdown(full_text)

    # Save assistant message
    if st.session_state.last_response:
        st.session_state.history.append({"role": "assistant", "content": st.session_state.last_response})

# Footer / Tips
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    tag("Tip")
    st.caption("Toggle Autorag to blend vector search.")
with col2:
    tag("Privacy")
    st.caption("Uploaded files stay in session memory only during runtime.")
with col3:
    tag("Citations")
    st.caption("RAG snippets are shown inline as [RAG #] in the answer body.")
