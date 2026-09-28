import chromadb

from source.ingest import load_documents
from source.chunk import chunk_documents
from source.embeddings import load_embedding_model, embed_documents


# Where Chroma stores its local database.
CHROMA_PATH = "data/chroma"

# Name of our vector collection.
COLLECTION_NAME = "personal_knowledge"


def get_collection():
    """
    Open the local Chroma database
    and return our document collection.
    """

    # PersistentClient saves data to disk.
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Create the collection if it does not exist.
    # Otherwise, load the existing one.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return collection


def store_chunks(collection, embedded_chunks: list[dict]) -> None:
    """
    Store embedded document chunks in Chroma.
    """

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in embedded_chunks:

        ids.append(
            chunk["chunk_id"]
        )

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        metadatas.append(
            {
                "source": chunk["source"],
                "chunk_number": chunk["chunk_number"],
            }
        )

    # upsert means:
    # insert if new
    # update if ID already exists
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_documents(
    collection,
    model,
    query: str,
    k: int = 3
) -> list[dict]:
    """
    Search the vector database for the
    most semantically relevant chunks.
    """

    # Convert the user's question into a vector.
    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search Chroma.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    search_results = []

    # Chroma returns nested lists because
    # it supports multiple queries at once.
    for i in range(len(results["ids"][0])):

        result = {
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }

        search_results.append(result)

    return search_results


if __name__ == "__main__":

    # -----------------------------
    # Build / load the knowledge base
    # -----------------------------

    documents = load_documents()

    chunks = chunk_documents(documents)

    model = load_embedding_model()

    embedded_chunks = embed_documents(
        model,
        chunks
    )

    collection = get_collection()

    store_chunks(
        collection,
        embedded_chunks
    )

    print()
    print(
        f"Stored {collection.count()} chunks in Chroma."
    )

    # -----------------------------
    # Test semantic search
    # -----------------------------

    query = "Which AWS service can migrate databases?"

    print()
    print(f"QUERY: {query}")
    print("=" * 70)

    results = search_documents(
        collection,
        model,
        query,
        k=3
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"RESULT #{rank}")
        print(
            f"Source: {result['metadata']['source']}"
        )
        print(
            f"Chunk: {result['metadata']['chunk_number']}"
        )
        print(
            f"Distance: {result['distance']:.4f}"
        )

        print("-" * 70)

        print(result["text"])