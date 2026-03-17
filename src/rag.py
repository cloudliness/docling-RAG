import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL, LMSTUDIO_BASE_URL, RETRIEVAL_K
from src.ingest import load_vectorstore

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided context from documents.
Use ONLY the context below to answer the question. If the context doesn't contain enough information, say so.
Include relevant details and be specific. When referencing information, mention which source document it came from.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        header = doc.metadata.get("header_1", "")
        prefix = f"[Source: {source}"
        if header:
            prefix += f" | Section: {header}"
        prefix += f" | Chunk {i}]"
        formatted.append(f"{prefix}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain():
    """Build the RAG chain connecting ChromaDB retriever to LM Studio LLM."""
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    llm = ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",  # LM Studio doesn't require a real key
        model=CHAT_MODEL,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def query(question: str) -> dict:
    """Run a RAG query and return the answer with source documents."""
    chain, retriever = create_rag_chain()

    # Get answer
    answer = chain.invoke(question)

    # Get source docs for reference
    source_docs = retriever.invoke(question)

    sources = []
    for doc in source_docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("header_1", ""),
            "content_preview": doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content,
        })

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    result = query(question)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  - {s['source']} | {s['section']}")
