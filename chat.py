# chat.py — Minimal Q&A over existing ChromaDB (clean UI, no debug box)
# Deps: streamlit, chromadb>=0.5.4, python-dotenv, openai>=1.30.0

import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# --- sqlite shim for Streamlit Cloud ---
# Ensures we use a modern sqlite (from pysqlite3-binary) even if system sqlite is old
try:
    import sys
    import pysqlite3  # provided by pysqlite3-binary
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # If pysqlite3-binary isn't available, we just proceed and let Chroma handle/raise
    pass
# ---------------------------------------

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI



# -------------------- Config --------------------
# -------------------- Config (Streamlit Secrets + .env fallback) --------------------
import os
from pathlib import Path

# 1) Try Streamlit secrets; if not available, use dotenv/OS env
try:
    import streamlit as st
    _SECRETS = dict(st.secrets)
except Exception:
    _SECRETS = {}

if not _SECRETS:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        load_dotenv(dotenv_path=env_path, override=True)
    except Exception:
        pass

def env_get(key: str, default=None):
    if key in _SECRETS:
        return _SECRETS[key]
    return os.getenv(key, default)

CHROMA_PATH     = Path(env_get("CHROMA_PATH", "./chroma_db")).resolve()
COLLECTION_NAME = env_get("COLLECTION_NAME", "Chat_Bot")
EMBED_MODEL     = env_get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL      = env_get("CHAT_MODEL", "gpt-4o-mini")
TOPK            = int(env_get("TOPK", 4))

from openai import OpenAI
OPENAI_API_KEY = env_get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (set in Streamlit Secrets or .env)")
client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------- Helpers --------------------
def compose_context(documents: List[str], metadatas: List[Dict[str, Any]], cap: int = 8000) -> str:
    out, used = [], 0
    for d, m in zip(documents, metadatas):
        src = m.get("file", m.get("source", "kb"))
        seg = f"\n---\nSource: {src}\n---\n{d}"
        if used + len(seg) > cap:
            break
        out.append(seg)
        used += len(seg)
    return "\n".join(out)

def answer_with_openai(api_key: str, model: str, system_prompt: str, user_msg: str) -> str:
    if not api_key:
        return "⚠️ Missing OPENAI_API_KEY. Set it in your .env."
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return (resp.choices[0].message.content or "").strip()

# -------------------- Chroma (cached) --------------------
@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    emb = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb
    )

# -------------------- UI --------------------
st.set_page_config(page_title="Chat Bot Q&A", page_icon="💬", layout="wide")
st.markdown(
    "<h1 style='margin-bottom:0'>Chat Bot Q&A</h1>"
    "<p style='color:#68707b'>Answers from your local ChromaDB.</p>",
    unsafe_allow_html=True,
)

# Optional: one-click cache clear if you changed CHROMA_PATH/COLLECTION_NAME
with st.sidebar:
    if st.button("♻️ Clear cache & reload"):
        st.cache_resource.clear()
        st.rerun()

# Connect to Chroma
try:
    collection = get_collection()
    st.caption(f"Connected • {CHROMA_PATH} • {COLLECTION_NAME} • Count: {collection.count()}")
except Exception as e:
    st.error("Couldn't open the Chroma collection. Check CHROMA_PATH / COLLECTION_NAME.")
    st.exception(e)
    st.stop()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

chat = st.container()
for m in st.session_state.history:
    with chat.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_query = st.chat_input("Ask a question…")
if not user_query:
    st.stop()

# Show user message
with chat.chat_message("user"):
    st.markdown(user_query)
st.session_state.history.append({"role": "user", "content": user_query})

# Retrieve + answer
with st.spinner("Thinking…"):
    try:
        q = collection.query(query_texts=[user_query], n_results=TOPK)
        docs = q.get("documents", [[]])[0]
        metas = q.get("metadatas", [[]])[0]
    except Exception as e:
        docs, metas = [], []
        st.error("Retrieval failed:")
        st.exception(e)

system_prompt = (
    "You are a helpful assistant that answers questions using only the provided document chunks"
    "When giving an answer: "
"- Extract the necessary information directly from the chunks. "
"- Cite the exact file/source you used in square brackets like [filename]. "
"- If the answer is not found in the provided chunks, Say no information is related to the query is found in the documents."
)
context = compose_context(docs, metas)
final_user = f"Question: {user_query}\n\nContext:\n{context}"

answer = answer_with_openai(OPENAI_API_KEY, CHAT_MODEL, system_prompt, final_user)

with chat.chat_message("assistant"):
    st.markdown(answer or "_(no answer)_")
    if docs:
        with st.expander("Sources", expanded=False):
            for d, m in zip(docs, metas):
                src = m.get("file", m.get("source", "kb"))
                preview = (d or "")[:800]
                ellipsis = "…" if d and len(d) > len(preview) else ""
                st.markdown(f"**{src}**\n\n{preview}{ellipsis}")

st.session_state.history.append({"role": "assistant", "content": answer or ""})
