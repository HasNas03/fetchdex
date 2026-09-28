from source.embeddings import load_embedding_model
from source.vector_search import get_collection, search_documents
from source.llm import generate_response


def build_context(search_results: list[dict]) -> str:
    """
    Convert retrieved chunks into numbered source blocks
    that the LLM can cite.
    """

    context_parts = []

    for index, result in enumerate(search_results, start=1):

        source = result["metadata"]["source"]
        chunk_number = result["metadata"]["chunk_number"]
        text = result["text"]

        context_part = (
            f"[SOURCE {index}]\n"
            f"Document: {source}\n"
            f"Chunk: {chunk_number}\n"
            f"Content:\n{text}"
        )

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)

def build_prompt(question: str, context: str) -> str:
    """
    Build a grounded prompt that requires citations.
    """

    return f"""
        You are a personal knowledge assistant.

        Answer the user's question using ONLY the provided context.

        Rules:
        - Do not use outside knowledge.
        - Do not invent facts.
        - If the answer is not supported by the context, say:
        "I don't know based on the provided documents."
        - Keep the answer concise.
        - Cite the supporting source using [SOURCE X].
        - Only cite sources that actually support the claim.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """.strip()

def extract_sources(search_results: list[dict]) -> list[dict]:
    """
    Convert retrieval results into clean source metadata
    for the final response.
    """

    sources = []

    for index, result in enumerate(search_results, start=1):

        source = {
            "citation": f"SOURCE {index}",
            "source": result["metadata"]["source"],
            "chunk_number": result["metadata"]["chunk_number"],
            "distance": result["distance"],
        }

        sources.append(source)

    return sources

def answer_question(
    question: str,
    k: int = 3
) -> dict:
    """
    Complete RAG pipeline with source tracking.
    """

    # Load local embedding model.
    model = load_embedding_model()

    # Load Chroma collection.
    collection = get_collection()

    # Retrieve relevant chunks.
    results = search_documents(
        collection=collection,
        model=model,
        query=question,
        k=k
    )

    # Build the context given to the LLM.
    context = build_context(results)

    # Build the grounded prompt.
    prompt = build_prompt(
        question=question,
        context=context
    )

    # Generate answer locally.
    answer = generate_response(prompt)

    # Create clean source metadata.
    sources = extract_sources(results)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": results,
    }

if __name__ == "__main__":

    question = (
        "Which AWS service can migrate databases?"
    )

    response = answer_question(question)

    print("\nQUESTION:")
    print(response["question"])

    print("\nANSWER:")
    print(response["answer"])

    print("\nSOURCES:")

    for source in response["sources"]:

        print(
            f"- [{source['citation']}] "
            f"{source['source']} "
            f"(chunk {source['chunk_number']})"
        )

    question = (
        "Which AWS service can migrate databases?"
    )

    response = answer_question(question)

    print("\nQUESTION:")
    print(response["question"])

    print("\nANSWER:")
    print(response["answer"])

    print("\nRETRIEVED CHUNKS:")

    for result in response["retrieved_chunks"]:

        print("-" * 70)

        print(
            f"Source: {result['metadata']['source']}"
        )

        print(
            f"Chunk: {result['metadata']['chunk_number']}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print()

        print(result["text"])