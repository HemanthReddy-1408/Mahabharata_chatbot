
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
# 🧘 Mahabharata Spiritual Guru: A RAG-based AI Chatbot Inspired by Krishna

## 📖 Project Overview

This project began with a powerful question —  
**"Can I make a chatbot that answers like Krishna?"**

Not just a typical Q&A bot, but a **spiritual guru** that offers:
- Context-aware answers  
- Philosophically rich insights  
- Emotionally intelligent guidance  

Rooted deeply in the wisdom of the **Mahabharata**, **Bhagavad Gita**, and classical Indian scriptures.

What started as a technical challenge quickly evolved into a journey — one that blended AI engineering with spiritual exploration.

---

## 🔧 Phase 1: Understanding the Soul of the Project

Before diving into code, I clarified the **purpose** of this assistant:

> 🕉️ It wasn’t just about answering questions — it had to be a guide, a companion, a modern Krishna.

### ✨ Key Design Goals

- **Spiritual empathy** — comfort users when they feel lost or confused  
- **Scripture-grounded answers** — draw only from Gita and Mahabharata, not generic AI responses  
- **Consistent tone** — mytho-philosophical, calm, and reflective  
- **Multilingual readiness** — able to understand English, Sanskrit, Hinglish  

---

## 📂 Phase 2: Building the RAG Pipeline

The assistant was built on a **Retrieval-Augmented Generation (RAG)** architecture with conversational memory.

### 🧱 Tech Stack

- **LangChain** — for orchestration and memory  
- **FAISS** — for efficient vector similarity search  
- **SentenceTransformers** — to generate high-quality embeddings  
- **Ollama** — to run local LLMs like Mistral or TinyLlama

### 🛠️ Key Components

- `preprocess_mahabharata.py` — Cleans and chunks the Mahabharata PDF  
- `embed_chunks.py` — Embeds chunks and stores them in FAISS  
- `rag_ask.py` — Handles user queries with context-aware generation

### 📚 What I Learned

- How to **chunk large philosophical texts** while maintaining coherence  
- Importance of **embedding model choice** for nuanced scripture  
- How to build **system prompts** that preserve tone and emotional depth

---

## 🔄 Phase 3: Interactivity and Emotional Connection

I wanted the bot to feel personal — like:

> “Don’t be afraid, I’m here with you. Krishna once told Arjuna...”

### 🤖 Features

- Emotionally-aware response templates  
- **LangChain’s ConversationBufferMemory** to keep dialogue flowing  
- A clean and simple **Streamlit UI** for user interaction

### 🧘 What I Learned

- Why **tone and emotional intelligence** are crucial in LLM prompting  
- How to balance **retrieved facts** with **creative generation**  
- The power of **session memory** in making bots feel truly alive

---

## 💡 Key Takeaways

### 👨‍💻 Technical

- End-to-end understanding of **RAG systems**: chunking, embedding, vector DBs, prompt design  
- Building modular, efficient, real-world pipelines with **LangChain + FAISS**  
- Practical use of conversational memory in a retrieval setting

### 🧠 Philosophical

- **AI without intention is just automation** — what you build it to say defines its value  
- Blending **dharmic wisdom** with modern AI makes timeless knowledge more accessible  
- The **soul of a chatbot** lies not in speed or accuracy — but in how it makes people feel

---

> *"In the midst of code, I found the Gita. In the silence of retrieval, I heard Krishna."*

---

