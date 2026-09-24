# ============================================================
# rag_pipeline.py
# Refactored from rag.ipynb — same logic, wrapped in functions
# ============================================================

import os
import re
import logging
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- CONFIG ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
LLM_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- PROMPT TEMPLATE (unchanged from your notebook) ----------
PROMPT = PromptTemplate(
    template="""You are an intelligent assistant that answers questions strictly based on the provided YouTube video transcript.

INSTRUCTIONS:
- Answer ONLY using the information from the context below.
- If the answer is not in the context, say: "I don't know "
- Do NOT make up facts, examples, or timestamps.
- Be concise but complete. Use bullet points for multi-part answers.
- If quoting the speaker, wrap the quote in quotation marks.
- Mention approximate timestamps if they appear in the context.

CONTEXT FROM VIDEO TRANSCRIPT:
{context}

USER QUESTION:
{question}

ANSWER:""",
    input_variables=["context", "question"],
)


# ---------- LLM ----------
def get_llm():
    """Create the Groq LLM client."""
    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE)


# ---------- URL PARSING ----------
def extract_video_id(url: str):
    """Extract 11-char video ID from any common YouTube URL."""
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None


# ---------- TRANSCRIPT ----------
def get_transcript(video_id: str):
    """Fetch transcript text for a given video ID. Returns None on failure."""
    try:
        yt_api = YouTubeTranscriptApi()
        transcripts = yt_api.fetch(video_id)
        text = " ".join(doc.text for doc in transcripts)
        logger.info(f"Fetched transcript for {video_id}: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"Transcript fetch failed for {video_id}: {e}")
        return None


# ---------- VECTOR STORE ----------
def build_vectorstore(transcript_text: str):
    """Split transcript, embed, and return a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript_text)
    logger.info(f"Split into {len(chunks)} chunks")

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embedding)
    logger.info("FAISS vector store built")
    return vector_store


# ---------- ASK ----------
def ask_question(vector_store, question: str) -> str:
    """Retrieve context, format prompt, call LLM, return answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)
    context = " ".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"question": question, "context": context})

    llm = get_llm()
    answer = llm.invoke(final_prompt)
    logger.info(f"Answered: {question[:60]}")
    return answer.content


# ---------- SUMMARIZE (optional) ----------
def summarize_video(vector_store) -> str:
    """Summarize the entire video in 5 bullet points."""
    docs = vector_store.similarity_search("main topic summary", k=10)
    context = " ".join(d.page_content for d in docs)

    summary_prompt = f"""Summarize this YouTube video in 5 clear bullet points.
Use only the information below.

CONTENT:
{context}

SUMMARY:"""

    llm = get_llm()
    return llm.invoke(summary_prompt).content


# ---------- QUICK TEST (run directly to verify) ----------
if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=0XStJayUrUY"
    vid = extract_video_id(test_url)
    print("Video ID:", vid)

    txt = get_transcript(vid)
    if txt:
        vs = build_vectorstore(txt)
        print("\n--- Q&A ---")
        print(ask_question(vs, "How to start the company?"))
        print("\n--- Summary ---")
        print(summarize_video(vs))