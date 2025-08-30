"""
StudyMate — Minimal end-to-end prototype (single-file) using Streamlit.
Features:
- Upload one or more PDFs
- Extract text with PyMuPDF (fitz)
- Chunk text for embeddings
- Create embeddings with SentenceTransformers
- Store / search with FAISS
- Generate final answer using IBM watsonx (Mixtral-8x7B-Instruct) via REST API (placeholder)

Run:
1. python -m venv venv
2. source venv/bin/activate   # (or venv\Scripts\activate on Windows)
3. pip install -r requirements.txt
4. streamlit run studymate_streamlit.py
"""

import os
import json
import textwrap
from typing import List, Tuple

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

# ----------------------- Config -----------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # compact & fast
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunks_metadata.json"

# Watsonx (Mixtral) placeholders — set these env vars before running
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_ENDPOINT = os.getenv("WATSONX_ENDPOINT", "")

# ----------------------- Utilities -----------------------

def extract_text_from_pdf_bytes(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text)"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append((i + 1, text))
    return pages


def chunk_text(pages: List[Tuple[int, str]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Produce chunks from pages, keeping reference to page numbers."""
    chunks = []
    for page_no, text in pages:
        text = text.replace("\n", " ")
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "page": page_no,
                    "text": chunk
                })
            if end >= len(text):
                break
            start = end - overlap
    return chunks


@st.cache_resource(show_spinner=False)
def get_embedding_model(name: str = EMBED_MODEL_NAME):
    return SentenceTransformer(name)


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_faiss(index, path: str):
    faiss.write_index(index, path)


def load_faiss(path: str):
    return faiss.read_index(path)


def call_watsonx_generate(prompt: str, max_tokens: int = 512) -> str:
    """Placeholder function for calling IBM watsonx Mixtral model.
    Replace with the authenticated request format required by your instance.
    """
    if not WATSONX_API_KEY or not WATSONX_ENDPOINT:
        return "[LLM not configured — set WATSONX_API_KEY and WATSONX_ENDPOINT environment variables]"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WATSONX_API_KEY}"
    }

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    try:
        resp = requests.post(WATSONX_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "text" in data:
            return data["text"]
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "")
        return json.dumps(data)
    except Exception as e:
        return f"[Error calling Watsonx: {str(e)}]"


def assemble_prompt(question: str, retrieved_chunks: List[dict]) -> str:
    """Create a prompt that provides context to the LLM and asks for a grounded answer."""
    context_texts = []
    for i, c in enumerate(retrieved_chunks):
        header = f"[Source {i+1} — page {c['page']}]\n"
        snippet = (c['text'][:800] + "...") if len(c['text']) > 800 else c['text']
        context_texts.append(header + snippet)

    context = "\n\n".join(context_texts)

    system = (
        "You are StudyMate, an AI assistant that answers student questions using only the provided document excerpts. "
        "If the answer cannot be found in the excerpts, say you don't know and suggest where to look or how to rephrase the question."
    )

    prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nINSTRUCTIONS: Provide a concise, student-friendly answer. Cite source numbers like [Source 1] when you use them."
    return prompt


# ----------------------- Streamlit App -----------------------

st.set_page_config(page_title="StudyMate — PDF Q&A", layout="wide")
st.title("StudyMate — AI PDF Q&A (Prototype)")

with st.sidebar:
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader("Choose one or more PDFs", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.header("2. Index Options")
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=3000, value=CHUNK_SIZE)
    overlap = st.number_input("Overlap (chars)", min_value=0, max_value=1000, value=CHUNK_OVERLAP)
    st.markdown("---")
    st.header("Watsonx settings")
    st.text_input("Watsonx Endpoint (env WATSONX_ENDPOINT)", value=WATSONX_ENDPOINT, key="w_endpoint")
    st.text_input("Watsonx API_KEY (env WATSONX_API_KEY)", value="***hidden***", key="w_key")

process_button = st.sidebar.button("Process & Build Index")

if process_button and uploaded_files:
    all_chunks = []
    for up in uploaded_files:
        st.sidebar.write(f"Processing: {up.name}")
        file_bytes = up.read()
        pages = extract_text_from_pdf_bytes(file_bytes)
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            c["source_file"] = up.name
        all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No text extracted. Try different PDFs or check if they are scanned images (OCR needed).")
    else:
        st.info(f"Created {len(all_chunks)} chunks. Building embeddings...")
        embed_model = get_embedding_model()
        texts = [c["text"] for c in all_chunks]
        embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        st.info("Building FAISS index...")
        index = build_faiss_index(embeddings)
        save_faiss(index, FAISS_INDEX_FILE)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        st.success("Index built and saved. You can now ask questions!")

# Load index if exists
index = None
chunks_metadata = None
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE):
    try:
        index = load_faiss(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            chunks_metadata = json.load(f)
    except Exception:
        index = None
        chunks_metadata = None

st.header("Ask a question")
question = st.text_input("Enter your question here:")
num_results = st.slider("Number of context chunks to retrieve", min_value=1, max_value=8, value=4)
ask_btn = st.button("Ask")

if ask_btn:
    if index is None or chunks_metadata is None:
        st.error("No index found. Upload PDFs and click 'Process & Build Index' first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        embed_model = get_embedding_model()
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        D, I = index.search(q_emb.astype(np.float32), k=num_results)
        retrieved = []
        for idx in I[0]:
            if idx < len(chunks_metadata):
                retrieved.append(chunks_metadata[idx])

        st.subheader("Retrieved Contexts")
        for i, r in enumerate(retrieved):
            with st.expander(f"Source {i+1} — {r.get('source_file','unknown')} (page {r['page']})"):
                st.write(textwrap.shorten(r["text"], width=1000))

        prompt = assemble_prompt(question, retrieved)
        with st.spinner("Generating answer from Watsonx..."):
            answer = call_watsonx_generate(prompt)

        st.markdown("---")
        st.subheader("Answer")
        st.write(answer)

        st.markdown("---")
        st.subheader("Citations")
        for i, r in enumerate(retrieved):
            st.write(f"[Source {i+1}] {r.get('source_file','unknown')} — page {r['page']}")

# ----------------------- Footer / Help -----------------------
st.markdown("---")
st.markdown("**Help / Notes:**")
st.markdown(
    "- If PDFs are scanned images, run OCR (e.g., Tesseract) before using this tool.\n"
    "- Increase chunk overlap or use a stronger SentenceTransformer model for better retrieval.\n"
    "- Configure Watsonx endpoint & API key via environment variables for real LLM answers.\n"
)
  
