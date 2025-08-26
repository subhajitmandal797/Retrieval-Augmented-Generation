# test_query.py
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv (dotenv_path=env_path, override=True)
CHROMA_PATH = Path(r"C:\Users\subhajitmandal\Documents\Retrieval-Augmented-Generation-main\Retrieval-Augmented-Generation\chroma_db")
COLLECTION_NAME = "Chat_Bot"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = "text-embedding-3-small"

emb = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBED_MODEL,
)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb)

print("Count:", col.count())
q = col.query(query_texts=["What is the document about?"], n_results=5)
for d, m in zip(q["documents"][0], q["metadatas"][0]):
    print("—", m.get("file"), "page", m.get("page"), "|", d[:120].replace("\n", " "), "…")
