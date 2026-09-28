from sentence_transformers import SentenceTransformer
from ingest import load_documents
from chunk import chunk_documents
from sentence_transformers.util import cos_sim

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """
    load local embedding model
    """
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    return model


def embed_documents(model: SentenceTransformer, chunks: list[dict]) -> list[dict]:
    """
    Convert every input chunk into an embedding vector
    """

    # 1. extract text from each chunk
    texts = [chunk["text"] for chunk in chunks]
    # 2. convert chunk texts into vectors
    embeddings = model.encode_document(texts, normalize_embeddings=True)
    # 3. attach each vector to its original chunk.
    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunk = {
            **chunk, 
            "embedding": embedding.tolist()
        }

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks

def test_similarity(model: SentenceTransformer) -> None:
    """
    Compare a query against two candidate documents.
    """

    query = "Which AWS service can migrate databases?"

    relevant_text = (
        "AWS DMS is a service used to migrate databases."
    )

    unrelated_text = (
        "FastAPI uses Pydantic for request validation."
    )

    # Embed the user question as a query.
    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True
    )

    # Embed candidate passages as documents.
    document_embeddings = model.encode_document(
        [
            relevant_text,
            unrelated_text
        ],
        normalize_embeddings=True
    )

    # Calculate semantic similarity.
    relevant_score = cos_sim(
        query_embedding,
        document_embeddings[0]
    ).item()

    unrelated_score = cos_sim(
        query_embedding,
        document_embeddings[1]
    ).item()

    print("\nSEMANTIC SIMILARITY TEST")
    print("-" * 50)

    print(f"Query: {query}\n")

    print(f"Relevant document score: {relevant_score:.3f}")
    print(f"Unrelated document score: {unrelated_score:.3f}")

if __name__ == "__main__":

    # 1. load docs
    documents = load_documents()

    # 2. chunk docs
    chunks = chunk_documents(documents)

    # 3. load embedding model
    model = load_embedding_model()

    # 4. generate embeddings
    embedded_chunks = embed_documents(model, chunks)

    print()
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedded chunks: {len(embedded_chunks)}")

    print()

    # Inspect the first embedded chunk.
    first_chunk = embedded_chunks[0]

    print("SOURCE:")
    print(first_chunk["source"])

    print()

    print("TEXT:")
    print(first_chunk["text"])

    print()

    print("EMBEDDING DIMENSIONS:")
    print(len(first_chunk["embedding"]))

    print()

    print("FIRST 10 VECTOR VALUES:")
    print(first_chunk["embedding"][:10])

    print()

    test_similarity(model)