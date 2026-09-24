# ============================================================
# app.py — Streamlit UI for YouTube RAG Chatbot
# ============================================================

import streamlit as st
from rag_pipeline import (
    extract_video_id,
    get_transcript,
    build_vectorstore,
    ask_question,
    summarize_video,
)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #f0f0f0;
    }

    /* Hero title */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #ff6b6b); }
        to   { filter: drop-shadow(0 0 20px #48dbfb); }
    }

    .hero-subtitle {
        text-align: center;
        color: #b8b8d1;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 1rem;
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.6rem 0;
        color: white;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .chat-bot {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.6rem 0;
        color: #e8e8f0;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #f0f0f0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #b8b8d1;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ---------- HERO HEADER ----------
st.markdown('<div class="hero-title">🎥 YouTube RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask anything about any YouTube video — powered by RAG + Groq LLM</div>',
    unsafe_allow_html=True,
)


# ---------- SESSION STATE ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "summary" not in st.session_state:
    st.session_state.summary = None


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Video Setup")
    url = st.text_input(
        "🔗 Paste YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
    )

    if st.button("🚀 Load Video", use_container_width=True):
        if not url:
            st.warning("Please paste a URL first.")
        else:
            video_id = extract_video_id(url)
            if not video_id:
                st.error("❌ Invalid YouTube URL.")
            else:
                with st.spinner("Fetching transcript & building index..."):
                    transcript = get_transcript(video_id)
                    if not transcript:
                        st.error("❌ No transcript available for this video.")
                    else:
                        st.session_state.vectorstore = build_vectorstore(transcript)
                        st.session_state.video_id = video_id
                        st.session_state.messages = []
                        st.session_state.summary = None
                        st.success("✅ Video loaded! Start asking questions.")

    st.markdown("---")
    st.markdown("### 📊 Status")
    if st.session_state.vectorstore is not None:
        st.markdown(f"**Video ID:** `{st.session_state.video_id}`")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        st.markdown(
            f"[▶️ Watch on YouTube](https://youtube.com/watch?v={st.session_state.video_id})"
        )
    else:
        st.info("No video loaded yet.")

    st.markdown("---")
    st.caption("Built with LangChain · FAISS · HuggingFace · Groq")


# ---------- MAIN TABS ----------
tab1, tab2 = st.tabs(["💬 Ask Questions", "📝 Summarize Video"])


# ---------- TAB 1: Q&A ----------
with tab1:
    if st.session_state.vectorstore is None:
        st.markdown(
            '<div class="glass-card">👈 Load a YouTube video from the sidebar to get started.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Render chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Input
        question = st.text_input(
            "Ask a question:",
            key="qa_input",
            placeholder="What is the main topic?",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            ask_btn = st.button("Ask 💡", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear Chat 🗑️", use_container_width=True)

        if clear_btn:
            st.session_state.messages = []
            st.rerun()

        if ask_btn and question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                answer = ask_question(st.session_state.vectorstore, question)
            st.session_state.messages.append({"role": "bot", "content": answer})
            st.rerun()


# ---------- TAB 2: SUMMARIZE ----------
with tab2:
    if st.session_state.vectorstore is None:
        st.markdown(
            '<div class="glass-card">👈 Load a video first to generate a summary.</div>',
            unsafe_allow_html=True,
        )
    else:
        if st.button("📝 Generate Summary", use_container_width=False):
            with st.spinner("Summarizing..."):
                st.session_state.summary = summarize_video(st.session_state.vectorstore)

        if st.session_state.summary:
            st.markdown(
                f'<div class="glass-card">{st.session_state.summary}</div>',
                unsafe_allow_html=True,
            )