# app.py — Minimal Q&A Chatbot over ChromaDB (no settings UI)

import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# -------- Config from env (or hardcode defaults below) --------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "growing_vegetables")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOPK = int(os.getenv("TOPK", "4"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -------- Helpers --------
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
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()

# -------- UI (plain chat) --------
st.set_page_config(page_title="Chat Bot Q&A", page_icon="💬", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>Chat Bot Q&A</h1><p style='color:#68707b'>Answers from your local ChromaDB.</p>", unsafe_allow_html=True)

# Connect to Chroma once
@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)

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
    "You are a helpful assistant that answers using only the provided KB excerpts. "
    "Cite each file/source you rely on in square brackets like [filename]. "
    "If the answer isn't in the excerpts, say you don't know."
)
context = compose_context(docs, metas)
final_user = f"Question: {user_query}\n\nContext:\n{context}"

answer = answer_with_openai(OPENAI_API_KEY, CHAT_MODEL, system_prompt, final_user)

with chat.chat_message("assistant"):
    st.markdown(answer)
    if docs:
        with st.expander("Sources", expanded=False):
            for d, m in zip(docs, metas):
                src = m.get("file", m.get("source", "kb"))
                st.markdown(f"**{src}**\n\n{d[:800]}{'…' if len(d) > 800 else ''}")

st.session_state.history.append({"role": "assistant", "content": answer})
