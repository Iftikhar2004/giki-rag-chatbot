import os
import numpy as np
import faiss
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

from ingest_utils import read_pdf, read_docx, read_txt, pages_to_chunks

# -----------------------------
# Config
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="GIKI Prospectus RAG Chatbot", page_icon="📘", layout="wide")

INDEX_PATH = "vector_store.index"
META_PATH = "vector_meta.pkl"

# ---- Custom CSS for Attractive UI ----
st.markdown("""
    <style>
    body { background: linear-gradient(to right, #1e3c72, #2a5298); font-family: 'Arial', sans-serif; }
    .navbar { background: rgba(255, 255, 255, 0.15); padding: 15px; border-radius: 12px; margin-bottom: 20px;
              text-align: center; color: white; font-size: 22px; font-weight: bold; }
    .chat-bubble { padding: 12px 18px; margin: 8px 0; border-radius: 18px; max-width: 70%; line-height: 1.5; font-size: 16px; }
    .user-bubble { background-color: #4CAF50; color: white; margin-left: auto; text-align: right; }
    .assistant-bubble { background-color: #f1f0f0; color: black; margin-right: auto; text-align: left; }
    .stButton button { background: linear-gradient(90deg, #667eea, #764ba2); color: white !important; border: none;
                      border-radius: 10px; padding: 12px 20px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }
    .stButton button:hover { background: linear-gradient(90deg, #6a11cb, #2575fc); transform: scale(1.05); }
    .stSlider > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Embedder Loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# -----------------------------
# Simple Vector Store
# -----------------------------
class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []

    def add(self, vecs: np.ndarray, meta):
        faiss.normalize_L2(vecs)
        self.index.add(vecs.astype("float32"))
        self.meta.extend(meta)

    def search(self, qvec: np.ndarray, k: int = 5):
        faiss.normalize_L2(qvec)
        D, I = self.index.search(qvec.astype("float32"), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((idx, float(score)))
        return results

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.meta, f)

    def load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.meta = pickle.load(f)
            return True
        return False

# -----------------------------
# RAG Utilities
# -----------------------------
def build_context(retrieved):
    lines = []
    for r in retrieved:
        tag = f"[Source: {r['source']} | Page: {r['page']}]"
        lines.append(f"{tag}\n{r['text']}")
    return "\n\n".join(lines)

def generate_answer(question, context, concise=True):
    prompt = f"""
    You are a helpful assistant for GIKI Prospectus Q&A.
    Question: {question}

    Context from prospectus:
    {context}

    Provide a {'short and to-the-point' if concise else 'detailed'} answer.
    Always stay factual. If unsure, say so.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

# -----------------------------
# Top Navbar
# -----------------------------
st.markdown("<div class='navbar'> GIKI Prospectus RAG Chatbot</div>", unsafe_allow_html=True)
st.caption("Upload prospectus → Build knowledge base → Ask questions with citations.")

# -----------------------------
# Sidebar Settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K Chunks to Retrieve", 3, 10, 5)
    chunk_size = st.slider("Chunk Size", 400, 1200, 800, step=50)
    overlap = st.slider("Chunk Overlap", 50, 300, 150, step=10)
    concise_mode = st.radio("Answer Style", ["Concise", "Detailed"], index=0)

# -----------------------------
# Session State
# -----------------------------
if "embedder" not in st.session_state:
    st.session_state.embedder = load_embedder()
if "store" not in st.session_state:
    dim = st.session_state.embedder.get_sentence_embedding_dimension()
    st.session_state.store = VectorStore(dim)
    st.session_state.store.load()  # load from disk if available
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("1) Upload your documents (max 3)")
files = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if files:
    if len(files) > 3:
        st.error(" Please upload at most 3 files.")
        files = files[:3]

    if st.button("⚡ Build knowledge base", type="primary"):
        pages_all = []
        for f in files:
            fname = f.name
            data = f.read()
            if fname.lower().endswith(".pdf"):
                pages = read_pdf(data, fname)
            elif fname.lower().endswith(".docx"):
                pages = read_docx(data, fname)
            else:
                pages = read_txt(data, fname)
            pages_all.extend(pages)

        chunks = pages_to_chunks(pages_all, chunk_size=chunk_size, overlap=overlap)
        texts = [c["text"] for c in chunks]
        embedder = st.session_state.embedder
        vecs = embedder.encode(texts, convert_to_numpy=True)
        st.session_state.store.add(vecs, chunks)
        st.session_state.store.save()
        st.success(f" Added {len(files)} file(s), {len(chunks)} chunks indexed. (Saved for future use)")

# -----------------------------
# Chat Section
# -----------------------------
st.subheader("2) Chat with Prospectus")

for role, content in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-bubble user-bubble'>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble assistant-bubble'>{content}</div>", unsafe_allow_html=True)

q = st.chat_input("Ask me something about GIKI...")
if q:
    st.session_state.chat.append(("user", q))

    if st.session_state.store.index.ntotal == 0:
        ans = " Please upload documents and click **Build knowledge base** first."
        retrieved = []
    else:
        embedder = st.session_state.embedder
        qvec = embedder.encode([q], convert_to_numpy=True)
        hits = st.session_state.store.search(qvec, k=top_k)
        retrieved = [st.session_state.store.meta[idx] for idx, score in hits]
        context = build_context(retrieved)
        ans = generate_answer(q, context, concise=(concise_mode == "Concise"))

    st.session_state.chat.append(("assistant", ans))
    st.markdown(f"<div class='chat-bubble assistant-bubble'>{ans}</div>", unsafe_allow_html=True)

    if retrieved:
        with st.expander(" Show sources"):
            for i, r in enumerate(retrieved, start=1):
                st.markdown(f"**{i}. {r['source']} — Page {r['page']}**\n\n{r['text'][:600]}…")

st.divider()
st.caption(" Tip: Concise mode gives short answers; switch to Detailed for longer explanations.")
