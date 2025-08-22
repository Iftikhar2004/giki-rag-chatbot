from typing import List, Dict
import io
from pypdf import PdfReader
import docx2txt
import re

# ----------- File Reading -----------

def read_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"text": clean_text(text), "page": i, "source": filename})
    return pages


def read_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path) or ""
    finally:
        os.remove(tmp_path)
    return [{"text": clean_text(text), "page": 1, "source": filename}]


def read_txt(file_bytes: bytes, filename: str) -> List[Dict]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"text": clean_text(text), "page": 1, "source": filename}]


# ----------- Cleaning & Chunking -----------

def clean_text(text: str) -> str:
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def pages_to_chunks(pages: List[Dict], chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    out = []
    for p in pages:
        for ch in chunk_text(p["text"], chunk_size=chunk_size, overlap=overlap):
            out.append({"text": ch, "page": p["page"], "source": p["source"]})
    return out
