import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to load FAISS index and chunks metadata
def load_index_and_chunks(index_path="maha_faiss.index", chunks_path="maha_chunks_metadata.json"):
    """
    Loads the FAISS index and the corresponding chunks (metadata).
    """
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

# Function to generate embeddings for a query
def generate_query_embedding(query, model):
    """
    Generates embeddings for a given query using a specified model.
    """
    return model.encode([query], convert_to_numpy=True)

# Function to get top-k similar chunks from FAISS
def get_top_k_chunks(query, index, chunks, model, k=3):
    """
    Retrieves the top-k most similar chunks to the query using the FAISS index.
    """
    query_vec = generate_query_embedding(query, model)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# Function to format the prompt for the LLM
def build_prompt(context_chunks, user_question):
    """
    Builds a structured prompt for the language model using context and the user's question.
    """
    context = "\n\n".join(context_chunks)
    return f"""You are a Mahabharata expert. Use the following context to answer the user's question.

Context:
{context}

Question: {user_question}

Answer:"""
