
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

This project started out with a simple question —  
**"Can I make a chatbot that answers like Krishna?"**  

Not just a typical Q&A bot, but a **spiritual guru** that gives:
- Context-aware answers
- Philosophically rich insights
- Emotionally intelligent guidance  
All rooted in the **wisdom of the Mahabharata**, **Bhagavad Gita**, and classical Indian scriptures.

What followed was an intense, incredibly rewarding journey into **LLMs**, **Retrieval-Augmented Generation (RAG)**, **fine-tuning**, **embeddings**, **vector databases** — and more importantly, **my own growth as an AI engineer and spiritual thinker**.

---

## 🔧 Phase 1: Understanding the Soul of the Project

Before writing a single line of code, I clarified the **purpose**:

> 🕉️ This assistant wasn't just about answering — it had to become a **spiritual companion**.

### ✨ Key Design Principles

- **Reflect spiritual empathy**  
  E.g., comforting users when they’re lonely, confused, or in pain.

- **Use the Mahabharata and Gita as the single source of truth**  
  No generic GPT-style responses allowed.

- **Maintain a mytho-philosophical tone**  
  Speak like how Krishna would counsel Arjuna — wise, calm, poetic.

- **Multilingual Capabilities**  
  Handle English, Sanskrit, or even Hinglish.

### 💡 Core Stack Needed

- **🧠 RAG** to ground answers in scripture
- **🛠️ QLoRA Fine-tuning** to give the LLM a "Krishna-like" voice
- **🧵 Conversational Memory** for genuine spiritual dialogue

---

## 📂 Phase 2: Building the RAG Pipeline

### 🧱 Stack Used

- **LangChain** — memory + orchestration  
- **FAISS** — fast vector search  
- **SentenceTransformers** — for embeddings  
- **Ollama** — to run local models like Mistral or TinyLlama

### 🛠️ Scripts Developed

- `preprocess_mahabharata.py` — Clean and chunk massive Mahabharata PDFs  
- `embed_chunks.py` — Embed chunks and index them into FAISS  
- `rag_ask.py` — Retrieve context + generate spiritual responses

### 📚 What I Learned

- How to **chunk large texts** while preserving coherence
- Why **embedding model choice** matters for abstract/philosophical data
- Crafting **system prompts** that respect sacred text tones

---

## 🔄 Phase 3: Interactivity and Emotional Intelligence

I wanted the bot to speak gently — like:

> “Don’t be afraid, I’m here with you. Krishna once told Arjuna...”

### 🤖 UX Features

- **Emotion-aware prompts** with spiritual tone
- **LangChain’s ConversationBufferMemory** for ongoing dialogue
- **Streamlit UI** for lightweight, interactive frontend

### 🧘 What I Learned

- Tone is everything in LLM prompting
- How to **blend factual scripture** with **creative generation**
- Why **memory systems** make the chatbot feel human

---

## 🔬 Phase 4: Fine-Tuning with QLoRA

This was the **hardest phase**, especially with:

> ⚙️ **RTX 3050 (4GB VRAM)**

### ⚒️ Solution Strategy

- Used **TinyLlama** and **Mistral-7B-Instruct** (as hardware allowed)
- Used **QLoRA + PEFT + bitsandbytes** for low-resource fine-tuning
- Built my **custom dataset** from scratch

### 🧾 Custom Dataset Highlights

- 100s of emotionally intelligent, spiritually grounded Q&A pairs
- Every response started with **“Krishna said...”** or **“My friend, listen to this story...”**
- Manually curated to reflect **compassion, wisdom, and relevance**

### 🎯 What I Learned

- Even **100–200 examples**, if crafted well, can **tune the soul of an LLM**
- The difference between **instruction-following** and **vibe-following**
- The joy of watching a model evolve into a **real virtual guru**

---

## 💡 Key Learnings

### 👨‍💻 Technical

- Deep understanding of **RAG**: chunking, embeddings, retrieval, prompt chaining
- Efficient fine-tuning with **QLoRA** under **VRAM constraints**
- LangChain memory, chaining, and prompt tools
- Clean, modular, and real-world ready LLM pipelines

### 🧠 Philosophical

- **AI without intention is just automation.** What matters is *what you make it say.*
- There’s beauty in **digitizing dharma** and making ancient wisdom accessible.
- The **future of chatbots isn’t speed — it’s soul.**

---

## 🚀 What’s Next?

- 🔊 **Speech-to-text** input for hands-free interaction  
- 📤 Let users **upload their life problems** and get scriptural-based responses  
- 🪷 Open-source the fine-tuned model on Hugging Face  
- 🌐 Launch a full-fledged **spiritual guidance web app** for the world  

---

## 🙏 Final Thoughts

If you're considering building something that **blends history, AI, and meaning** —  
I **highly recommend it**.

This wasn’t just an NLP project.  
It was a **yajna** — a sacred offering of effort, devotion, and learning.

---

> *"In the midst of code, I found the Gita. In the silence of fine-tuning, I heard Krishna."*

---

