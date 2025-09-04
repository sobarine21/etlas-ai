# app.py — Streamlit Chatbot with Gemini + Cloudflare Autorag (AI Search)
# ---------------------------------------------------------------
# Features
# - Real multi-turn chat UI (chat_input + message bubbles)
# - Toggle retrieval via Cloudflare Autorag AI Search
# - Upload files (PDF/TXT/MD/DOCX) and ask Q&A grounded on their content
# - Streams Gemini responses token-by-token
# - System prompts, temperature slider, history clear, and chat export
# - Uses Streamlit Secrets for all keys and IDs
#
# Required secrets in .streamlit/secrets.toml
# -------------------------------------------
# GEMINI_API_KEY = "your_gemini_api_key"
# CLOUDFLARE_EMAIL = "you@example.com"
# CLOUDFLARE_API_KEY = "your_cf_global_api_key"
# CLOUDFLARE_ACCOUNT_ID = "your_cf_account_id"
# CLOUDFLARE_AUTORAG_ID = "your_autorag_rag_id"
#
# Suggested requirements.txt
# --------------------------
# streamlit
# google-genai>=0.3.0
# requests
# PyPDF2
# python-docx
#
# Run locally
# -----------
#   streamlit run app.py
# ---------------------------------------------------------------

from __future__ import annotations
import os
import io
import json
import time
import textwrap
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

# --- Page config
st.set_page_config(
    page_title="Gemini + Autorag Chat",
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
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
CF_EMAIL = st.secrets.get("CLOUDFLARE_EMAIL", "")
CF_API_KEY = st.secrets.get("CLOUDFLARE_API_KEY", "")
CF_ACCOUNT_ID = st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_AUTORAG_ID = st.secrets.get("CLOUDFLARE_AUTORAG_ID", "")

# --- Import Gemini client lazily so app can still render UI without key
def get_gemini_client():
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)

SUPPORTED_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash-8b",
]

# --- Session State
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []  # [{"role":"user|assistant", "content":"..."}]
if "files_ctx" not in st.session_state:
    st.session_state.files_ctx = ""  # concatenated extracted text from uploads
if "raw_files" not in st.session_state:
    st.session_state.raw_files = []  # keep references to UploadedFiles
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

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
    """Trim huge context. Gemini handles long input, but keep prompts snappy."""
    if len(txt) <= max_chars:
        return txt
    head = txt[: max_chars // 2]
    tail = txt[-max_chars // 2 :]
    return head + "\n\n…\n\n" + tail


def call_cloudflare_autorag(query: str, *, max_snippets: int = 5) -> Dict[str, Any]:
    """Query Cloudflare Autorag AI Search endpoint and return JSON response.
    Expecting the endpoint that returns results for a 'query'. If the service returns
    passages/documents, we'll format them below.
    """
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
    # Try common shapes; adjust as Autorag evolves
    # e.g., data = {"result": {"hits": [{"text":"...","source":"..."}, ...]}}
    try:
        result = data.get("result") or data
        hits = result.get("hits") or result.get("results") or []
        for h in hits[:max_snippets]:
            txt = h.get("text") or h.get("chunk") or h.get("content") or ""
            src = h.get("source") or h.get("metadata", {}).get("source") or h.get("url") or ""
            if txt:
                snippets.append({"text": txt, "source": src})
    except Exception:
        pass

    return {"raw": data, "snippets": snippets}


def build_augmented_prompt(user_query: str, *, use_files: bool, use_autorag: bool) -> str:
    ctx_parts = []

    if use_files and st.session_state.files_ctx:
        ctx_parts.append("User-uploaded context (excerpt):\n" + chunk_text(st.session_state.files_ctx, 12000))

    if use_autorag:
        rag = call_cloudflare_autorag(user_query)
        st.session_state.last_autorag = rag
        if rag.get("error"):
            ctx_parts.append(f"[Autorag error] {rag['error']}")
        else:
            snips = rag.get("snippets", [])
            if snips:
                formatted = []
                for i, s in enumerate(snips, 1):
                    src = f" (source: {s['source']})" if s.get("source") else ""
                    formatted.append(f"[{i}] {s['text'][:1500]}{src}")
                ctx_parts.append("Cloudflare Autorag retrieval:\n" + "\n\n".join(formatted))
            else:
                ctx_parts.append("Cloudflare Autorag returned no snippets.")

    system_instructions = textwrap.dedent(
        """
        You are a helpful AI assistant inside a Streamlit app. Follow the rules:
        - If context is provided below, cite it concisely in-line as [RAG #].
        - Prefer factual, step-by-step answers. If uncertain, say so.
        - Keep code blocks minimal and runnable where possible.
        - When answering about uploaded files, ground your answer strictly in that content.
        - If a user asks general questions, answer broadly, but still prefer accuracy.
        """
    ).strip()

    context_blob = ("\n\n" + "\n\n".join(ctx_parts)) if ctx_parts else ""

    return (
        system_instructions
        + "\n\nUser question: "
        + user_query
        + context_blob
    )


def stream_gemini_response(prompt: str, *, model: str, temperature: float = 0.4):
    # Build content as per user's provided sample code (with streaming)
    from google.genai import types
    client = get_gemini_client()

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    # Gemini streaming
    yield from client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    )


# --- SIDEBAR CONFIG
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Configure model + retrieval + context")

    model = st.selectbox("Gemini model", SUPPORTED_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    use_autorag = st.toggle("Use Cloudflare Autorag AI Search", value=True)
    use_files = st.toggle("Use uploaded files as context", value=True)

    with st.expander("Cloudflare Autorag credentials (read-only)"):
        st.text_input("Account ID", value=CF_ACCOUNT_ID, disabled=True)
        st.text_input("RAG ID", value=CF_AUTORAG_ID, disabled=True)
        st.text_input("Email", value=CF_EMAIL, disabled=True)
        obscured = (CF_API_KEY[:4] + "***" + CF_API_KEY[-4:]) if CF_API_KEY else ""
        st.text_input("API Key", value=obscured, disabled=True)

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
st.title("🤖 Gemini + Autorag Chatbot")

st.markdown(
    "Ask questions, upload files, and optionally augment with Cloudflare Autorag retrieval."
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

        # Guardrails: check keys
        if not GEMINI_API_KEY:
            st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
        else:
            # Build augmented prompt
            augmented = build_augmented_prompt(prompt, use_files=use_files, use_autorag=use_autorag)

            try:
                # Stream tokens
                for chunk in stream_gemini_response(augmented, model=model, temperature=temperature):
                    text = getattr(chunk, "text", None)
                    if not text:
                        continue
                    full_text += text
                    placeholder.markdown(full_text)
                st.session_state.last_response = full_text
            except Exception as e:
                st.error(f"Gemini error: {e}")
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
    st.caption("Toggle Autorag to blend vector search with Gemini.")
with col2:
    tag("Privacy")
    st.caption("Uploaded files stay in session memory only during runtime.")
with col3:
    tag("Citations")
    st.caption("RAG snippets are shown inline as [RAG #] in the answer body.")
