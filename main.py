import os
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

app = FastAPI(
    title="AI Document Intelligence API",
    version="0.1.0",
    description="Upload PDFs and search them semantically using embeddings.",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)
document_store: list[dict[str, Any]] = []


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Simple chunking strategy based on words."""
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def extract_pdf_chunks(file_path: str) -> list[dict[str, Any]]:
    reader = PdfReader(file_path)
    chunks: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            chunks.append({
                "page": page_number,
                "text": chunk,
            })

    return chunks


@app.get("/")
async def home() -> dict[str, str]:
    return {"message": "AI Document Intelligence API running!"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "documents_loaded": len(document_store),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    chunks = extract_pdf_chunks(file_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found in the uploaded PDF.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.cpu().numpy()

    global document_store
    document_store = chunks

    return {
        "filename": file.filename,
        "chunks_processed": len(chunks),
        "embedding_dimensions": embeddings.shape[1] if len(embeddings) > 0 else 0,
    }


@app.post("/search")
async def search_document(query: str) -> dict[str, Any]:
    if not document_store:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet.")

    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    corpus_embeddings = np.vstack([chunk["embedding"] for chunk in document_store]).astype(np.float32)
    corpus_tensor = torch.tensor(corpus_embeddings)

    similarities = util.cos_sim(query_embedding, corpus_tensor)[0]
    top_results = similarities.topk(k=min(5, len(document_store)))

    results: list[dict[str, Any]] = []
    for index, score in zip(top_results.indices.tolist(), top_results.values.tolist()):
        chunk = document_store[index]
        results.append({
            "page": chunk["page"],
            "score": round(float(score), 4),
            "text": chunk["text"],
        })

    return {
        "query": query,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
