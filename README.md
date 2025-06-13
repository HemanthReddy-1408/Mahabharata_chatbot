# Mahabharata RAG QA System

This project implements a Retrieval-Augmented Generation (RAG) pipeline that uses the Mahabharata and related texts as a knowledge base. It is designed to provide deep, contextual, and spiritual answers to life questions. The pipeline uses local language models via Ollama, FAISS for vector search, and LangChain for orchestration.

---

## Project Structure

.
├── .cache/ # LangChain cache
│ └── langchain_cache.db

├── data/ # Core data assets
│ ├── Mahabharata_Unabridged.pdf # Primary knowledge source
│ ├── maha_chunks.json # Text split into vector-searchable chunks
│ ├── maha_chunks_metadata.json # Metadata for chunk provenance
│ ├── maha_faiss.index # Vector index (FAISS)

├── llms/
│ └── ollama_llm.py # Local LLM wrapper using LangChain and Ollama

├── path_to_cache_file/
│ └── langchain_cache.db # LangChain cache duplicate (optional)

├── utils/
│ ├── rag_helpers.py # RAG utilities: search, prompt building, retrieval
│ ├── streamlit_helpers.py # Utilities for the Streamlit frontend

├── .gitignore
├── README.md # Project documentation
├── embed_chunks.py # Embedding + indexing pipeline
├── mahabharata_rag_app.py # Streamlit interface
├── preprocess_mahabharata.py # PDF extraction and text cleaning
├── rag_ask.py # Command-line RAG interface
├── requirements.txt # Python dependencies

yaml
Copy
Edit

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mahabharata-rag.git
cd mahabharata-rag
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Make sure to install faiss-cpu, sentence-transformers, langchain, PyPDF2, ollama, and streamlit.

3. Configure Environment (Optional)
Create a .env file if needed:

ini
Copy
Edit
PDF_PATH=data/Mahabharata_Unabridged.pdf
CHUNKS_PATH=data/maha_chunks.json
CHUNKS_METADATA_PATH=data/maha_chunks_metadata.json
FAISS_INDEX_PATH=data/maha_faiss.index
EMBEDDING_MODEL=all-MiniLM-L6-v2
OLLAMA_MODEL=mistral
Pipeline Overview
Step 1: Preprocess PDF
Clean and extract chunks from the Mahabharata source.

bash
Copy
Edit
python preprocess_mahabharata.py
Step 2: Create FAISS Index
Generate embeddings for the chunks and build a FAISS vector index.

bash
Copy
Edit
python embed_chunks.py
Step 3: Query via CLI
Ask questions directly using your terminal.

bash
Copy
Edit
python rag_ask.py
Step 4: Streamlit UI (Optional)
Launch the graphical interface to chat with the assistant.

bash
Copy
Edit
streamlit run mahabharata_rag_app.py
Module Descriptions
Module/File	Description
preprocess_mahabharata.py	Loads PDF and splits into clean text chunks
embed_chunks.py	Embeds chunks using sentence-transformers and stores them in FAISS
rag_ask.py	CLI application to chat with the system
mahabharata_rag_app.py	Streamlit frontend
ollama_llm.py	LangChain-compatible wrapper for local LLMs via Ollama
rag_helpers.py	Chunk retrieval, prompt creation, and response generation logic
streamlit_helpers.py	Handles UI logic and response formatting for Streamlit

Future Work
Integrate LoRA/QLoRA fine-tuning using emotional-spiritual datasets

Add Whisper for voice input

Support Hinglish and Sanskrit shloka recognition

Add LangChain memory for multi-turn dialogue context

Fine-tune Mistral or TinyLlama to respond in philosophical tone

