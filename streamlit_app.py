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
            src = h.get("source") or (h.get("metadata") or {}).get("source") or h.get("url") or ""
            if txt:
                snippets.append({"text": txt, "source": src, "raw_hit": h})
    except Exception:
        pass

    return {"raw": data, "snippets": snippets}


def prepare_contents_for_gemini(user_query: str, *, use_files: bool, use_autorag: bool) -> List[Any]:
    """
    Build structured contents (list of types.Content) for Gemini streaming API.
    This includes:
      - system instruction content (explicit rules + citation instruction)
      - context contents (uploaded files excerpt and Autorag snippets, each labelled)
      - user content (the question)
    Returns a list of types.Content objects (but typing as Any to avoid import at top-level).
    """
    from google.genai import types

    # Base system instruction, but include explicit direction to cite RAG results.
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

    contents: List[Any] = []
    # Add system content
    contents.append(
        types.Content(
            role="system",
            parts=[types.Part.from_text(text=system_instructions)]
        )
    )

    # Add uploaded files context as a single system/context block (if requested)
    if use_files and st.session_state.files_ctx:
        files_excerpt = chunk_text(st.session_state.files_ctx, 12000)
        file_block = "=== Uploaded files context ===\n\n" + files_excerpt
        contents.append(
            types.Content(
                role="system",
                parts=[types.Part.from_text(text=file_block)]
            )
        )

    # Add Autorag snippets as separate numbered block if requested
    if use_autorag:
        # Call Autorag here so the RAG result is available to include as a context block
        rag = call_cloudflare_autorag(user_query)
        st.session_state.last_autorag = rag
        if rag.get("error"):
            rag_block = f"[Autorag error] {rag['error']}"
            contents.append(
                types.Content(
                    role="system",
                    parts=[types.Part.from_text(text=rag_block)]
                )
            )
        else:
            snips = rag.get("snippets", [])
            if snips:
                # build numbered snippet block with explicit [RAG n] labels
                formatted_snips = []
                for i, s in enumerate(snips, start=1):
                    src = f" (source: {s['source']})" if s.get("source") else ""
                    # ensure snippet is trimmed reasonably
                    snippet_text = s['text']
                    snippet_text = snippet_text if len(snippet_text) <= 4000 else snippet_text[:4000] + "\n\n[TRUNCATED]"
                    formatted_snips.append(f"[RAG {i}]{src}\n{snippet_text}")
                rag_block = "=== Cloudflare Autorag retrieval (numbered) ===\n\n" + "\n\n---\n\n".join(formatted_snips)
                contents.append(
                    types.Content(
                        role="system",
                        parts=[types.Part.from_text(text=rag_block)]
                    )
                )
            else:
                contents.append(
                    types.Content(
                        role="system",
                        parts=[types.Part.from_text(text="Cloudflare Autorag returned no snippets.")]
                    )
                )

    # Finally add the user question (as user role)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)]
        )
    )

    return contents


def stream_gemini_response(contents: List[Any], *, model: str, temperature: float = 0.4):
    """
    Stream Gemini response for the provided `contents` list (list of types.Content).
    Yields the chunks from the streaming generator so the UI can show incremental text.
    """
    from google.genai import types
    client = get_gemini_client()

    # Build a GenerateContentConfig similar to the user's sample
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    # Gemini streaming using structured contents
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

        # Guardrails: check keys
        if not GEMINI_API_KEY:
            st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
        else:
            # Build structured contents for Gemini that include Autorag context explicitly
            try:
                contents = prepare_contents_for_gemini(prompt, use_files=use_files, use_autorag=use_autorag)
            except Exception as e:
                st.error(f"Failed to prepare contents: {e}")
                contents = None

            if contents is not None:
                try:
                    # Stream tokens
                    for chunk in stream_gemini_response(contents, model=model, temperature=temperature):
                        # Each chunk may have attributes; prefer chunk.text
                        text = getattr(chunk, "text", None)
                        if not text:
                            # sometimes chunk.delta or chunk.content may exist
                            text = getattr(chunk, "delta", None) or getattr(chunk, "content", None) or ""
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
