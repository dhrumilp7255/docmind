"""
query.py
--------
Interactive terminal chatbot that answers questions
using your indexed documents via RAG.

Run after ingest.py:
    python query.py
"""

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

FAISS_INDEX_DIR = "faiss_index"
TOP_K_CHUNKS = 5  # Number of document chunks to retrieve per query


# Custom prompt that injects retrieved chunks as context
PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I couldn't find that in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    """Load the saved FAISS index from disk."""
    if not os.path.exists(FAISS_INDEX_DIR):
        print("❌ No FAISS index found. Please run 'python ingest.py' first.")
        exit(1)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def build_qa_chain(vectorstore):
    """Build the RetrievalQA chain with custom prompt."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def print_sources(source_docs):
    """Print the source chunks used to generate the answer."""
    print("\n  📄 Sources used:")
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        label = f"{source} (page {page + 1})" if page != "" else source
        if label not in seen:
            print(f"    - {label}")
            seen.add(label)


def main():
    print("=" * 50)
    print("  RAG Document Q&A")
    print("=" * 50)
    print("  Loading your knowledge base...")

    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    print("  ✅ Ready! Type your question below.")
    print("  Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            source_docs = result["source_documents"]

            print(f"\nAssistant: {answer}")
            print_sources(source_docs)
            print()

        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
