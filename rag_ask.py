import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

# === Load environment variables ===
load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
CHUNKS_METADATA_PATH = os.getenv("CHUNKS_METADATA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# === Enable LangChain Caching ===
set_llm_cache(InMemoryCache())

# === Load FAISS Index and Metadata ===
print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# === Load Embedding Model ===
print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# === Load Ollama LLM via LangChain ===
print(f"[INFO] Initializing LLM with model: {OLLAMA_MODEL}")
llm = Ollama(model=OLLAMA_MODEL)

# === Get Top-k Similar Chunks ===
def get_top_k_chunks(query, k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# === Prompt Builder ===
def build_prompt(context_chunks, user_question):
    context = "\n\n".join(context_chunks)
    return f"""You are a Mahabharata expert. Use the following context to answer the user's question.

Context:
{context}

Question: {user_question}

Answer:"""

# === Ask the LLM ===
def ask_ollama(prompt):
    return llm.invoke(prompt)

# === Ask Loop ===
def main():
    print("\nWelcome to the Mahabharata RAG QA System")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            print("👋 Exiting... Jai Shree Krishna!")
            break

        top_chunks = get_top_k_chunks(query)
        prompt = build_prompt(top_chunks, query)

        print("\n[CONTEXT USED]")
        for i, chunk in enumerate(top_chunks):
            print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...\n")

        print("[ANSWERING...]\n")
        answer = ask_ollama(prompt)
        print(answer.strip())

if __name__ == "__main__":
    main()
