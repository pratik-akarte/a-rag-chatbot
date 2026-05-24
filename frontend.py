import streamlit as st
from backend import process_pdf, get_answer

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Doc-Chat — RAG Assistant",
    page_icon="📄",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d0d0d; color: #e8e8e8; }
h1, h2, h3 { font-family: 'Space Mono', monospace; color: #f0f0f0; }

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-ready { background:#0f3d1f; color:#4ade80; border:1px solid #166534; }

.chat-user {
    background: #1a1a2e;
    border-left: 3px solid #6366f1;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-size: 0.95rem;
}
.chat-assistant {
    background: #161616;
    border-left: 3px solid #4ade80;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-size: 0.95rem;
}

div[data-testid="stTextInput"] input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

.stButton > button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
}
hr { border-color: #2a2a2a !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
if "retriever_chain" not in st.session_state:
    st.session_state.retriever_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# ── Header ────────────────────────────────────────────────────
st.markdown("## 📄 Doc-Chat")
st.markdown("<p style='color:#888; margin-top:-0.8rem; font-size:0.9rem;'>RAG-powered document assistant</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Upload Section ────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{uploaded_file.name}** — `{round(uploaded_file.size / 1024, 1)} KB`")
    with col2:
        if st.button("Process →"):
            with st.spinner("Building RAG pipeline..."):
                chain, info = process_pdf(uploaded_file.read(), uploaded_file.name)
                st.session_state.retriever_chain = chain
                st.session_state.doc_info = info
                st.session_state.chat_history = []
            st.rerun()

# ── Status Bar ────────────────────────────────────────────────
if st.session_state.doc_info:
    info = st.session_state.doc_info
    st.markdown(
        f'<span class="status-badge badge-ready">✓ READY</span>&nbsp;&nbsp;'
        f'<span style="color:#666; font-size:0.8rem;">'
        f'{info["filename"]} &middot; {info["pages"]} pages &middot; {info["chunks"]} chunks'
        f'</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

# ── Chat Interface ────────────────────────────────────────────
if st.session_state.retriever_chain:

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    query = st.text_input(
        "Ask something...",
        key="query_input",
        label_visibility="collapsed",
        placeholder="Ask something about the document..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("Ask →")
    with col2:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask and query.strip():
        with st.spinner("Thinking..."):
            answer = get_answer(st.session_state.retriever_chain, query)

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

elif not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:#444;">
        <div style="font-size:3rem; margin-bottom:1rem;">📂</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem;">
            Upload a PDF to get started
        </div>
    </div>
    """, unsafe_allow_html=True)