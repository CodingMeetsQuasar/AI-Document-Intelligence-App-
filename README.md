# AI Document Intelligence App

This project is a lightweight document intelligence API built with FastAPI and SentenceTransformers.

It lets you:
- upload a PDF document,
- extract text from it,
- split the text into chunks,
- embed each chunk using a sentence embedding model,
- search the document semantically by asking questions in natural language.

## Features

- PDF upload endpoint
- Text extraction using `pypdf`
- Chunked document processing
- Semantic similarity search using embeddings
- Local in-memory index for the current uploaded document

## Quick start

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .
   ```

2. Run the API:

   ```bash
   uvicorn main:app --reload
   ```

3. Upload a PDF:

   ```bash
   curl -X POST "http://localhost:8000/upload" -F "file=@your-document.pdf"
   ```

4. Search it:

   ```bash
   curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query":"What does this document say about compliance?"}'
   ```

## Notes

This is a backend-first prototype. It is intentionally simple and does not yet include a frontend UI, persistent database, or multi-document indexing.
