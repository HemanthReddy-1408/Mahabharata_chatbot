
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
📜 My Journey Building a Mahabharata Spiritual Guru RAG Assistant
This project started out with a simple question — "Can I make a chatbot that answers like Krishna?" Not just a typical Q&A bot, but a spiritual guru that gives context-aware, philosophically rich, and emotionally intelligent responses rooted in the wisdom of the Mahabharata, the Bhagavad Gita, and classical Indian scriptures. What followed was an intense, incredibly rewarding journey into LLMs, Retrieval-Augmented Generation, fine-tuning, embeddings, and vector databases — and more importantly, into my own growth as an AI engineer and spiritual thinker.

🔧 Phase 1: Understanding the Soul of the Project
Before writing a single line of code, I had to clarify what this assistant was supposed to be. It wasn't just about answering questions — it had to:

Reflect spiritual empathy (e.g., comforting users when they’re lonely or lost)

Use Mahabharata and Gita as source of truth, not some general GPT vibes

Maintain a consistent, mytho-philosophical tone — like how Krishna would counsel Arjuna

Possibly speak in English, Sanskrit, or even Hinglish

That’s when I realized: retrieval is not enough. I needed a combo of:

🧠 RAG to ground responses in scripture

🛠️ QLoRA fine-tuning to teach the LLM to speak like a guru

🧵 Memory to continue the conversation like a spiritual dialogue

📂 Phase 2: Building the RAG Pipeline (Retrieval-Augmented Generation)
I used:

LangChain for memory and orchestration

FAISS for fast vector search

SentenceTransformers to create embeddings

Ollama for running local LLMs like Mistral or TinyLlama

I also wrote:

preprocess_mahabharata.py to clean and chunk my massive Mahabharata PDF

embed_chunks.py to embed the cleaned chunks and index them in FAISS

rag_ask.py to build prompts and generate answers with retrieved context

What I learned here:

How to chunk massive PDFs and ensure coherence across overlapping passages

Why embeddings matter — and how different models (e.g., all-MiniLM, paraphrase-MPNet) perform for philosophical texts

How to craft system prompts that preserve tone, especially when dealing with sacred texts

🔄 Phase 3: Interactivity and Empathy
This is where I took it personally. I didn’t want the bot to just dump scripture. It had to talk like:

"Don’t be afraid, I’m here with you. Krishna once told Arjuna..."

So, I designed prompts to start with emotionally intelligent hooks, and used:

Custom prompt builders

LangChain’s ConversationBufferMemory to maintain thread context

Interactive UI with Streamlit

Through this, I learned:

The importance of tone in LLM prompting

How to balance retrieved content and generation creativity

Why even spiritual bots need session memory for genuine connection

🔬 Phase 4: Fine-Tuning with QLoRA
The biggest challenge was VRAM. I had an RTX 3050 with 4GB VRAM — not exactly a supercomputer.

So I chose:

Mistral-7B-Instruct (when possible)

TinyLlama for faster iteration

QLoRA + PEFT + bitsandbytes to fine-tune efficiently on custom datasets

I built my dataset from scratch:

Curated 100s of non-duplicate, emotionally rich Q&A pairs

Made sure each answer started with something like "Krishna said..." or "My friend, let me tell you a story from the Mahabharata..."

This taught me:

The true power of instruction-tuning — not just facts, but vibes

Why datasets need to feel human, not just look syntactically right

How even 100-200 examples, if crafted well, can bend the LLM’s tone significantly

💡 What I Learned (Technically + Philosophically)
Technical Learnings:

Deep understanding of RAG components: chunking, embedding, vector search, prompt chaining

Fine-tuning techniques like QLoRA and LoRA under resource constraints

LangChain’s memory systems and LLM tool orchestration

How to write clean, modular pipelines for real-world LLM systems

Personal/Philosophical Learnings:

AI without intention is just automation. What matters is what you make it say.

There’s power in digitizing dharma — making ancient wisdom more accessible.

The future of chatbots isn’t speed — it’s soul.

🚀 What’s Next?
Adding speech-to-text for hands-free use

Letting users upload their problems and get scriptural solutions

Open-sourcing the fine-tuned model on Hugging Face

Maybe even launching a web app for anyone seeking guidance

If you're thinking about building a project that goes beyond just code — that connects history, culture, and AI — I can’t recommend this kind of work enough.

This was not just an NLP project.

It was a yajna of learning, coding, and inner growth.

---

