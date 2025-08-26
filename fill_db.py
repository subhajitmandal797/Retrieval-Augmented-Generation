# index_pdfs_to_chroma.py
# pip install chromadb langchain-community langchain-text-splitters python-dotenv openai sentence-transformers pypdf

import os
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------- Config --------
env_path = Path(__file__).parent / ".env"
load_dotenv (dotenv_path=env_path, override=True)

DATA_PATH = Path(os.getenv("DATA_PATH", "data")).resolve()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db")).resolve()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Chat_Bot")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # OpenAI
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers

# -------- Embedding function (pick one) --------
emb_fn = None
if OPENAI_API_KEY:
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )
else:
    # Local CPU embedding (no API key); slower but works offline
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=LOCAL_EMBED_MODEL
    )

# -------- Chroma client/collection --------
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn,  # IMPORTANT
)

print(f"DB: {CHROMA_PATH}")
print(f"Collection: {COLLECTION_NAME} (pre-count={collection.count()})")

# -------- Load PDFs --------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")

loader = PyPDFDirectoryLoader(str(DATA_PATH))  # loads all PDFs under DATA_PATH
raw_documents = loader.load()

# -------- Split to chunks --------
# Your 300-char chunks are very small (~75 tokens). Use ~800–1200 chars for better retrieval.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # ~300 tokens (roughly)
    chunk_overlap=200,   # keeps context continuity
    length_function=len,
    is_separator_regex=False,
)

chunks = splitter.split_documents(raw_documents)
print(f"Loaded pages: {len(raw_documents)}  ->  Chunks: {len(chunks)}")

# -------- Prepare rows --------
def make_id(text: str, meta: dict) -> str:
    # stable ID from file path + page + chunk index + hash
    base = f"{meta.get('source','')}-{meta.get('page',-1)}-{meta.get('chunk',-1)}"
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{base}-{h}"

documents, metadatas, ids = [], [], []

# Some loaders put file path in metadata['source'] or 'file_path'; normalize:
def normalize_meta(m: dict):
    out = dict(m or {})
    path = out.get("source") or out.get("file_path") or ""
    out["path"] = str(path)
    out["file"] = Path(path).name if path else out.get("file", "unknown.pdf")
    out.setdefault("page", out.get("page", -1))
    return out

# Group chunks per file so we can assign a running chunk index per page/file
running_index = {}
for d in chunks:
    m = normalize_meta(d.metadata)
    key = (m.get("path"), m.get("page", -1))
    running_index[key] = running_index.get(key, -1) + 1
    m["chunk"] = running_index[key]

    text = d.page_content or ""
    if not text.strip():
        continue

    documents.append(text)
    metadatas.append({"file": m["file"], "path": m["path"], "page": m["page"], "chunk": m["chunk"]})
    ids.append(make_id(text, metadatas[-1]))

# -------- Upsert to Chroma --------
# upsert() is fine (idempotent by id). Use add() if you want errors on duplicates.
if documents:
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Upserted {len(documents)} chunks.")
else:
    print("No non-empty chunks to index.")

print(f"Collection count: {collection.count()}")
print("Done.")
