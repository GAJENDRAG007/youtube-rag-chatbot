# 🎥 YouTube RAG Chatbot

> Ask anything about any YouTube video — powered by Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

A conversational AI application that lets you **ask questions about any YouTube video** and get grounded, context-aware answers with zero hallucination risk.

The app fetches the video transcript, splits it into semantic chunks, embeds them using HuggingFace embeddings, stores them in a FAISS vector index, and uses a Groq-hosted LLM to generate answers strictly from the retrieved context.

---

## ✨ Features

- 🎨 **Modern glassmorphism UI** built with Streamlit
- 💬 **Chat-style Q&A** with message history
- 📝 **One-click video summarization** in 5 bullet points
- 🚀 **Fast retrieval** with FAISS + `all-MiniLM-L6-v2` embeddings
- ⚡ **Sub-second LLM responses** powered by Groq
- 🛡️ **Grounded answers** — the bot refuses to answer if the info isn't in the video
- 🔗 **Auto URL parsing** — supports `watch`, `youtu.be`, `shorts`, and `embed` formats
- 🧹 **Clean modular code** — UI separated from RAG pipeline logic

---

## 🏗️ Architecture

![Architecture](assets/architecture.png)

**Pipeline flow:**

1. **Ingestion** — YouTube transcript fetched via `youtube-transcript-api`
2. **Chunking** — `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
3. **Embedding** — `sentence-transformers/all-MiniLM-L6-v2`
4. **Indexing** — FAISS vector store
5. **Retrieval** — Top-K = 4 similarity search per query
6. **Augmentation** — Grounded prompt template with context + question
7. **Generation** — Groq LLM (`openai/gpt-oss-20b`, temp = 0.3)

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Streamlit (custom CSS theme) |
| **Orchestration** | LangChain |
| **Transcript** | youtube-transcript-api |
| **Embeddings** | HuggingFace `sentence-transformers` |
| **Vector Store** | FAISS |
| **LLM** | Groq (`openai/gpt-oss-20b`) |
| **Config** | python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

### 2. Create and activate the environment

```bash
conda create --name venv python=3.11 -y
conda activate venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Copy the template and paste your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---





---

## 📁 Project Structure

```
youtube-rag-chatbot/
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Core RAG logic (importable)
├── rag.ipynb               # Original experiments & tuning
├── requirements.txt        # Python dependencies
├── .env.example            # Env var template
├── .gitignore
├── README.md
└── assets/                 # Diagrams + screenshots
```

---

## 🧠 Design Decisions

**Why RAG instead of fine-tuning?**
RAG is cheaper, faster to iterate, and keeps answers grounded in the actual video content — no hallucinations, no retraining needed.

**Why FAISS?**
It's a lightweight, in-memory vector index that runs on a laptop with no server setup. Perfect for a single-video use case.

**Why `all-MiniLM-L6-v2`?**
It's a 22M-parameter model that's fast on CPU and produces high-quality semantic embeddings for English text.

**Why Groq?**
Groq's LPU inference delivers sub-second latency on open-source models, essential for a chat-style UX — and its free tier is generous.

**Why the strict prompt?**
The prompt explicitly forbids making up facts. If the answer isn't in the retrieved context, the bot says "I don't know." This is the foundation of trustworthy RAG.

---

## 🔮 Future Improvements

- [ ] Hybrid retrieval (BM25 + vector) for higher recall
- [ ] Reranking with a cross-encoder for precision
- [ ] Timestamp-grounded source attribution
- [ ] Multi-video comparison mode
- [ ] Response caching to reduce LLM calls
- [ ] Evaluation harness (Recall@K, faithfulness)

---

## 📜 License

MIT — free to use, modify, and share.

---

## 🙌 Acknowledgements

Built with [LangChain](https://python.langchain.com/), [Streamlit](https://streamlit.io/), [FAISS](https://github.com/facebookresearch/faiss), [HuggingFace](https://huggingface.co/), and [Groq](https://groq.com/).