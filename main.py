import io
import os
import chromadb
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

DB_DIR = "chroma_db"
os.makedirs(DB_DIR, exist_ok=True)
chroma_client = chromabd.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="pdf_documents")
document_chunks = []


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


@app.get("/")
def home():
    return {"message": "AI Document Intelligence API running!"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_chunks

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    reader = PdfReader(file_path)

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        page_chunks = chunk_text(text)

        for chunk in page_chunks:
            chunks.append({
                "page": page_number,
                "text": chunk
            })

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode_document(
        texts,
        convert_to_tensor=True
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    document_chunks = chunks

    return {
        "filename": file.filename,
        "number_of_chunks": len(chunks),
        "embedding_dimensions": embeddings.shape[1] if len(embeddings) > 0 else 0
    }


@app.post("/search")
async def search_document(query: str):
    if not document_chunks:
        return {
            "error": "No document has been uploaded yet."
        }

    query_embedding = model.encode_query(
        query,
        convert_to_tensor=True
    )

    corpus_embeddings = [chunk["embedding"] for chunk in document_chunks]

    search_results = util.semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=3
    )[0]

    results = []

    for result in search_results:
        chunk = document_chunks[result["corpus_id"]]

        results.append({
            "page": chunk["page"],
            "score": float(result["score"]),
            "text": chunk["text"]
        })

    return {
        "query": query,
        "results": results
    }