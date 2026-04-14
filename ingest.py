"""
ingest.py
---------
Loads all documents from the /documents folder (PDF, TXT, DOCX),
chunks them into overlapping segments, generates embeddings,
and saves the FAISS index locally.

Run this once (or whenever you add new documents):
    python ingest.py
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCUMENTS_DIR = "documents"
FAISS_INDEX_DIR = "faiss_index"

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(directory: str):
    """Walk through the documents folder and load all supported files."""
    all_docs = []
    files = os.listdir(directory)

    if not files:
        print("⚠️  No files found in the documents/ folder. Please add some files and run again.")
        return []

    for filename in files:
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"  Skipping unsupported file: {filename}")
            continue

        filepath = os.path.join(directory, filename)
        loader_class = SUPPORTED_EXTENSIONS[ext]
        print(f"  Loading: {filename}")

        try:
            loader = loader_class(filepath)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ❌ Failed to load {filename}: {e}")

    return all_docs


def chunk_documents(documents):
    """Split documents into overlapping chunks for better context retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # ~500 characters per chunk
        chunk_overlap=100,     # 100 character overlap between chunks
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\n  Total chunks created: {len(chunks)}")
    return chunks


def build_faiss_index(chunks):
    """Generate embeddings and build the FAISS vector store."""
    print("\n  Generating embeddings (this may take a moment)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"  ✅ FAISS index saved to '{FAISS_INDEX_DIR}/'")
    return vectorstore


def main():
    print("=" * 50)
    print("  RAG Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load documents
    print(f"\n[1/3] Loading documents from '{DOCUMENTS_DIR}/'...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        return
    print(f"  Loaded {len(documents)} document page(s)")

    # Step 2: Chunk documents
    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3: Build FAISS index
    print("\n[3/3] Building FAISS index...")
    build_faiss_index(chunks)

    print("\n✅ Ingestion complete! You can now run: python query.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
