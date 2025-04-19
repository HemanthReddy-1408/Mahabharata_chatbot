import streamlit as st
import json
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import the helper functions from utils
from utils.rag_helpers import get_top_k_chunks, build_prompt
from utils.streamlit_helpers import initialize_chat_history, save_chat_history

# Import the updated langchain-ollama package
from langchain_ollama import OllamaLLM
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# === Load environment variables ===
load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
CHUNKS_METADATA_PATH = os.getenv("CHUNKS_METADATA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
DB_PATH = os.getenv("DB_PATH", ".cache/langchain_cache.db")  # SQLite DB file path

# Ensure the cache directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# === Enable LangChain Caching with SQLite ===
set_llm_cache(SQLiteCache(DB_PATH))  # Use SQLiteCache for persistent storage

# === Load FAISS & Chunks ===
@st.cache_resource
def load_index_and_chunks():
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks
    except Exception as e:
        st.error(f"Error loading FAISS index or chunks: {e}")
        return None, None

# === Load Sentence Embedder ===
@st.cache_resource
def load_embedder():
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# === LangChain LLM Wrapper ===
@st.cache_resource
def load_llm():
    try:
        return OllamaLLM(model=OLLAMA_MODEL)
    except Exception as e:
        st.error(f"Error initializing Ollama model: {e}")
        return None

# === Streamlit Dark Theme ===
st.set_page_config(page_title="Mahabharata QA - RAG", layout="wide")

# Custom Dark Theme Config
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #121212;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    .stTextInput input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
    }
    .stButton button {
        background-color: #6c3483;
        color: white;
    }
    .stMarkdown, .stText {
        color: #ffffff;
    }
    .stSelectbox, .stTextInput {
        background-color: #333333;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Initialize chat history ===
initialize_chat_history()

# === Layout with two columns: Left for history, Center for input/output ===
col1, col2 = st.columns([2, 5])  # 2 parts for history, 5 parts for input/output

# Left column: Display the chat history (Only questions)
with col1:
    st.subheader("🗨️ Chat History")
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown("---")

# Right column: Input box for questions and displaying answers
with col2:
    st.title("Mahabharata Question Answering (RAG)")
    st.markdown("Ask any question about the Mahabharata and get contextual answers powered by FAISS + Ollama.")

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Thinking like Vyasa..."):
            index, chunks = load_index_and_chunks()
            embedder = load_embedder()
            llm = load_llm()

            # Ensure successful resource loading
            if index is None or chunks is None or embedder is None or llm is None:
                st.error("Failed to load necessary resources. Please check logs and try again.")
            else:
                # Get the top-k chunks for context
                top_chunks = get_top_k_chunks(query, index, chunks, embedder)  # Using the helper function
                
                # Build the prompt for the model
                prompt = build_prompt(top_chunks, query)  # Using the helper function
                
                # Get the answer from the LangChain LLM model
                answer = llm.invoke(prompt).strip()  # Calling the LangChain LLM model

                # Save to chat history
                st.session_state.chat_history.append({"question": query, "answer": answer})

                # Display the answer in the center
                st.markdown(f"**Oracle:** {answer}")

# === Clear and Save Chat History Buttons ===
if st.button("Clear Chat"):
    st.session_state.chat_history = []

if st.button("Save Chat History"):
    save_chat_history(st.session_state.chat_history)  # Using the helper function
