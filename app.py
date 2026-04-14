"""
app.py
------
Flask web server that connects the UI with the RAG pipeline.
Handles document uploads, ingestion, and Q&A queries.

Run with:
    python app.py
Then open: http://localhost:5000
"""

import os
import shutil
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__, static_folder="static")

DOCUMENTS_DIR = "documents"
FAISS_INDEX_DIR = "faiss_index"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}

PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I couldn't find that in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Global vectorstore cache
vectorstore_cache = None


def get_vectorstore():
    global vectorstore_cache
    if vectorstore_cache:
        return vectorstore_cache
    if not os.path.exists(FAISS_INDEX_DIR):
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore_cache = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore_cache


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore_cache
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    uploaded = []
    skipped = []

    for file in files:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(file.filename)
            continue
        filename = secure_filename(file.filename)
        file.save(os.path.join(DOCUMENTS_DIR, filename))
        uploaded.append(filename)

    if not uploaded:
        return jsonify({"error": "No supported files uploaded. Use PDF, TXT, or DOCX."}), 400

    # Ingest uploaded documents
    try:
        all_docs = []
        for filename in os.listdir(DOCUMENTS_DIR):
            ext = os.path.splitext(filename)[-1].lower()
            if ext not in LOADER_MAP:
                continue
            loader = LOADER_MAP[ext](os.path.join(DOCUMENTS_DIR, filename))
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)
        vectorstore_cache = vectorstore

        return jsonify({
            "message": f"Successfully indexed {len(uploaded)} file(s) into {len(chunks)} chunks.",
            "uploaded": uploaded,
            "skipped": skipped,
            "chunks": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    vectorstore = get_vectorstore()
    if not vectorstore:
        return jsonify({"error": "No documents indexed yet. Please upload documents first."}), 400

    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = []
        seen = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            label = f"{os.path.basename(source)}" + (f" (page {page + 1})" if page != "" else "")
            if label not in seen:
                sources.append(label)
                seen.add(label)

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/documents", methods=["GET"])
def list_documents():
    files = []
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            ext = os.path.splitext(f)[-1].lower()
            if ext in ALLOWED_EXTENSIONS:
                size = os.path.getsize(os.path.join(DOCUMENTS_DIR, f))
                files.append({"name": f, "size": round(size / 1024, 1)})
    return jsonify({"documents": files})


@app.route("/clear", methods=["POST"])
def clear():
    global vectorstore_cache
    try:
        if os.path.exists(DOCUMENTS_DIR):
            shutil.rmtree(DOCUMENTS_DIR)
            os.makedirs(DOCUMENTS_DIR)
        if os.path.exists(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR)
        vectorstore_cache = None
        return jsonify({"message": "All documents cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
