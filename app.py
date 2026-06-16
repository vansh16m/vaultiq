import streamlit as st
from dotenv import load_dotenv
from chain import load_qa_chain

load_dotenv()
import os
from ingest import build_vectorstore

if not os.path.exists("vectorstore/index.faiss"):
    with st.spinner("Building knowledge base from bank reports — this takes 2-3 minutes on first run..."):
        build_vectorstore()

st.set_page_config(
    page_title="VaultIQ",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif; }
html, body, [class*="css"] { background-color: #212121; color: #ECECEC; }
.stApp { background-color: #212121; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2F2F2F !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
section[data-testid="stSidebar"] .block-container { padding: 16px !important; }

/* All sidebar buttons */
.stButton > button {
    background: #2A2A2A !important;
    border: 1px solid #3F3F3F !important;
    border-radius: 8px !important;
    color: #ECECEC !important;
    font-size: 12px !important;
    text-align: left !important;
    padding: 9px 12px !important;
    width: 100% !important;
    white-space: normal !important;
    height: auto !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    background: #333 !important;
    border-color: #10A37F !important;
}
.new-chat > button {
    background: transparent !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    margin-bottom: 8px !important;
}

/* Messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    max-width: 720px;
    margin: 0 auto;
    padding: 4px 0 !important;
}
[data-testid="stChatMessage"] p {
    font-size: 15px !important;
    line-height: 1.75 !important;
    color: #ECECEC !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #10A37F, #0D8A6C) !important;
    border-radius: 6px !important;
}

/* Input bar — centred */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 28px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(720px, calc(100vw - 300px)) !important;
    background: transparent !important;
    border: none !important;
    z-index: 999 !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #2F2F2F !important;
    border: 1px solid #3F3F3F !important;
    border-radius: 14px !important;
    color: #ECECEC !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #10A37F !important;
    box-shadow: 0 0 0 2px rgba(16,163,127,0.15) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #2A2A2A !important;
    border: 1px solid #3F3F3F !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    color: #8A8A8A !important;
}
.streamlit-expanderContent {
    background: #2A2A2A !important;
    border: 1px solid #3F3F3F !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* Source tag */
.src-tag {
    display: inline-block; font-size: 11px; color: #8A8A8A;
    background: #333; border: 1px solid #3F3F3F;
    border-radius: 5px; padding: 2px 8px; margin: 2px 3px;
}

.stSpinner > div { border-top-color: #10A37F !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #3F3F3F; border-radius: 3px; }

.sidebar-label {
    font-size: 11px; color: #8A8A8A;
    text-transform: uppercase; letter-spacing: .07em;
    margin: 16px 0 6px; padding: 0 2px;
}

/* Bottom disclaimer — fixed like Claude */
.disclaimer-bar {
    position: fixed;
    bottom: 0;
    left: 260px;
    right: 0;
    text-align: center;
    font-size: 11px;
    color: #555;
    padding: 6px 0 8px;
    background: #212121;
    z-index: 998;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 VaultIQ")

    # New conversation
    st.markdown('<div class="new-chat">', unsafe_allow_html=True)
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        if "chain" in st.session_state:
            del st.session_state.chain
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Suggested questions
    st.markdown('<div class="sidebar-label">Suggested Questions</div>',
                unsafe_allow_html=True)
    suggestions = [
        "What was RBC's net income in 2024?",
        "Compare RBC and TD's capital ratios",
        "How did the HSBC acquisition affect RBC?",
        "What credit risks do Canadian banks face?",
        "Which Big 5 bank has the highest ROE?",
        "What is TD's dividend per share?",
        "How are rising rates affecting bank margins?",
        "What is OSFI's capital requirement?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state.pending_prompt = s
            st.rerun()

    # Stats
    st.markdown("""
    <div style="display:flex;gap:6px;margin:16px 0">
        <div style="flex:1;background:#2A2A2A;border-radius:8px;padding:8px 6px;text-align:center">
            <div style="font-size:13px;font-weight:500;color:#10A37F">900+</div>
            <div style="font-size:10px;color:#8A8A8A;margin-top:2px">Pages</div>
        </div>
        <div style="flex:1;background:#2A2A2A;border-radius:8px;padding:8px 6px;text-align:center">
            <div style="font-size:13px;font-weight:500;color:#10A37F">5K+</div>
            <div style="font-size:10px;color:#8A8A8A;margin-top:2px">Chunks</div>
        </div>
        <div style="flex:1;background:#2A2A2A;border-radius:8px;padding:8px 6px;text-align:center">
            <div style="font-size:13px;font-weight:500;color:#10A37F">5</div>
            <div style="font-size:10px;color:#8A8A8A;margin-top:2px">Banks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model info — at the very bottom
    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#ECECEC;padding:3px 2px">⚡ LLaMA 3.3 70B via Groq</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#8A8A8A;padding:3px 2px">🔍 FAISS vector search</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#8A8A8A;padding:3px 2px">🧠 Conversational memory</div>', unsafe_allow_html=True)

# ── Load chain ────────────────────────────────────────────
if "chain" not in st.session_state:
    with st.spinner("Loading VaultIQ..."):
        st.session_state.chain = load_qa_chain()

# ── Welcome screen ────────────────────────────────────────
if not st.session_state.messages and st.session_state.pending_prompt is None:
    st.markdown("""
    <div style="max-width:680px;margin:120px auto 0;padding:0 20px;text-align:center">
        <div style="width:56px;height:56px;
        background:linear-gradient(135deg,#10A37F,#0D8A6C);
        border-radius:16px;font-size:26px;display:flex;
        align-items:center;justify-content:center;margin:0 auto 20px">🏦</div>
        <div style="font-size:30px;font-weight:600;color:#ECECEC;margin-bottom:10px">
            How can I help you today?
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.markdown(f'<span class="src-tag">{src}</span>',
                                unsafe_allow_html=True)

# ── Handle prompt ─────────────────────────────────────────
def handle_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing reports..."):
            result = st.session_state.chain.invoke({"question": prompt})
            answer = result["answer"]
            sources = list(set([
                f"{doc.metadata.get('source','?').replace('docs/','').replace('.pdf','')} · p.{doc.metadata.get('page','?')}"
                for doc in result["source_documents"]
            ]))
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for src in sorted(sources):
                    st.markdown(f'<span class="src-tag">{src}</span>',
                                unsafe_allow_html=True)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    handle_prompt(prompt)

if prompt := st.chat_input("Ask Anything"):
    handle_prompt(prompt)

# ── Bottom disclaimer — fixed like Claude ─────────────────
st.markdown("""
<div class="disclaimer-bar">
    VaultIQ · Answers grounded in source documents · Not financial advice
</div>
""", unsafe_allow_html=True)
