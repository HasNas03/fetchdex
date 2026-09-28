from embeddings import load_embedding_model
from vector_search import get_collection, search_documents
from llm import generate_response


def build_context(search_results: list[dict]) -> str:
    """
    Combine retrieved chunks into one block of context
    for the LLM.
    """

    context_parts = []

    for result in search_results:

        source = result["metadata"]["source"]
        chunk_number = result["metadata"]["chunk_number"]
        text = result["text"]

        context_part = (
            f"Source: {source}, Chunk: {chunk_number}\n"
            f"{text}"
        )

        context_parts.append(context_part)

    # Separate chunks clearly.
    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """
    Create a grounded RAG prompt.
    """

    return f"""
You are a personal knowledge assistant.

Answer the user's question using ONLY the context below.

Rules:
- Do not use outside knowledge.
- If the answer is not supported by the context, say:
  "I don't know based on the provided documents."
- Keep the answer concise.
- Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()


def answer_question(
    question: str,
    k: int = 3
) -> dict:
    """
    Full RAG pipeline.

    1. Load embedding model
    2. Load vector database
    3. Retrieve relevant chunks
    4. Build context
    5. Build prompt
    6. Ask local LLM
    """

    # Load our local embedding model.
    model = load_embedding_model()

    # Load existing Chroma collection.
    collection = get_collection()

    # Retrieve semantically relevant chunks.
    results = search_documents(
        collection=collection,
        model=model,
        query=question,
        k=k
    )

    # Convert retrieved chunks into context.
    context = build_context(results)

    # Build the prompt for the LLM.
    prompt = build_prompt(
        question=question,
        context=context
    )

    # Ask local Ollama model.
    answer = generate_response(prompt)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": results
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