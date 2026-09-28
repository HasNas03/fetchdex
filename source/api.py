from fastapi import FastAPI, Query
from pydantic import BaseModel

from source.ingest import load_documents
from source.embeddings import load_embedding_model
from source.vector_store import get_collection, search_documents
from source.rag import answer_question


# Create FastAPI app.
app = FastAPI(
    title="Personal Knowledge Agent",
    description="Local RAG API for searching and asking questions about personal documents.",
    version="1.0.0"
)


# Load shared resources once when the app starts.
embedding_model = load_embedding_model()
collection = get_collection()


class AskRequest(BaseModel):
    """
    Request body for POST /ask.
    """

    question: str
    k: int = 3


@app.get("/health")
def health():
    """
    Simple health-check endpoint.
    """

    return {
        "status": "ok"
    }


@app.get("/documents")
def get_documents():
    """
    Return the names of loaded local documents.
    """

    documents = load_documents()

    return {
        "count": len(documents),
        "documents": [
            document["source"]
            for document in documents
        ]
    }


@app.get("/search")
def search(
    q: str = Query(
        ...,
        min_length=1,
        description="Semantic search query"
    ),
    k: int = Query(
        3,
        ge=1,
        le=10,
        description="Number of results to return"
    )
):
    """
    Perform semantic search without asking the LLM.
    """

    results = search_documents(
        collection=collection,
        model=embedding_model,
        query=q,
        k=k
    )

    return {
        "query": q,
        "results": results
    }


@app.post("/ask")
def ask(request: AskRequest):
    """
    Run the complete RAG pipeline.
    """

    response = answer_question(
        question=request.question,
        k=request.k
    )

    return response