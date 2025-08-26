import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# setting the environment

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables")


user_query = input("Hello How may I help you?\n\n")

results = collection.query(
    query_texts=[user_query],
    n_results=4
)

print(results['documents'])
#print(results['metadatas'])

client = OpenAI()

system_prompt = """
"You are a helpful assistant that answers using only the provided KB excerpts. "
"Cite each file you rely on in square brackets like [filename]. If the answer isn't in the excerpts, say so."
--------------------
The data:
"""+str(results['documents'])+"""
"""

#print(system_prompt)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
)

print("\n\n---------------------\n\n")

print(response.choices[0].message.content)