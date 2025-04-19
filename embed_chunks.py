import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env variables
load_dotenv()

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/maha_chunks.json")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/maha_faiss.index")
CHUNKS_METADATA_PATH = os.getenv("CHUNKS_METADATA_PATH", "data/maha_chunks_metadata.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Step 1: Load the chunks
if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"[INFO] Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

# Step 2: Load the sentence transformer model
print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# Step 3: Generate embeddings
print("[INFO] Generating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# Step 4: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"[SUCCESS] FAISS index created with {index.ntotal} vectors.")

# Step 5: Save FAISS index and metadata
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"[INFO] FAISS index saved to: {FAISS_INDEX_PATH}")

with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"[INFO] Chunk metadata saved to: {CHUNKS_METADATA_PATH}")
print("Vector DB creation completed successfully.")
