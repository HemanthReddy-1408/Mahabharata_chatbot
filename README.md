
# Mahabharata RAG QA System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using the Mahabharata epic as the core knowledge source. It leverages local language models (via Ollama), vector search (via FAISS), and orchestration with LangChain to create a robust Question-Answering system.

---

## 📁 Project Structure

```
.
├── .cache/
│   └── langchain_cache.db              # LangChain's local cache DB

├── data/
│   ├── Mahabharata_Unabridged.pdf      # Complete Mahabharata source
│   ├── maha_chunks.json                # Processed chunks for embedding
│   ├── maha_chunks_metadata.json       # Metadata for provenance/source tracking
│   ├── maha_faiss.index                # FAISS vector index file

├── llms/
│   └── ollama_llm.py                   # Wrapper for LangChain LLM interface using Ollama

├── path_to_cache_file/
│   └── langchain_cache.db              # Reference to LangChain cache path

├── utils/
│   ├── rag_helpers.py                  # RAG pipeline helper functions
│   ├── streamlit_helpers.py            # Streamlit UI helpers

├── .gitignore                          # Git ignore rules
├── README.md                           # Project documentation (this file)
├── embed_chunks.py                     # Script to embed text chunks into FAISS
├── mahabharata_rag_app.py              # Streamlit web application
├── preprocess_mahabharata.py           # PDF parsing, cleaning, and chunking logic
├── rag_ask.py                          # CLI script to ask questions from Mahabharata
├── requirements.txt                    # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mahabharata-rag.git
cd mahabharata-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

You can set custom paths via a `.env` file (optional). Default paths are hardcoded in scripts.

---

## Pipeline Workflow

### Step 1: Preprocess Mahabharata PDF

```bash
python preprocess_mahabharata.py
```

- Loads the PDF
- Extracts text
- Chunks text into manageable pieces

### Step 2: Embed Chunks and Build Index

```bash
python embed_chunks.py
```

- Converts text chunks to embeddings using Sentence Transformers
- Stores them in FAISS index

### Step 3: Query via CLI

```bash
python rag_ask.py
```

- Accepts user question
- Retrieves top-k similar chunks using FAISS
- Feeds context and question to Ollama LLM via LangChain
- Returns final response

### Step 4: Launch the Streamlit UI

```bash
streamlit run mahabharata_rag_app.py
```

- Simple web-based interface
- Uses same retrieval and generation logic as CLI

---

## Code Overview

| File | Description |
|------|-------------|
| `preprocess_mahabharata.py` | Reads PDF and cleans/splits text |
| `embed_chunks.py` | Embeds text into FAISS |
| `rag_ask.py` | Command-line interface for Q&A |
| `mahabharata_rag_app.py` | Streamlit-based UI |
| `ollama_llm.py` | Local LLM (Mistral) wrapper via Ollama |
| `rag_helpers.py` | Core logic for retrieval and prompt building |
| `streamlit_helpers.py` | UI formatting and streamlit helpers |

---

##  Roadmap

- Add memory-based chat experience
- Voice input via Whisper
- Multilingual support (Sanskrit, Hindi)
- LoRA/QLoRA fine-tuning pipeline

---

