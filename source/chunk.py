from source.ingest import load_documents

def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    """
    Split text into overlapping chunks of words.

    chunk_size: Maximum number of words in each chunk.
    overlap: Number of words repeated between consecutive chunks.
    """
    # TODO: decide on best chunking divide, e.g., words/sentences/paragraphs/etc.

    # chunk store
    chunks = []
    # convert text into individual words
    words = text.split()
    
    # start at first word
    start = 0

    while start < len(words):
        # calculate chunk end
        end = start + chunk_size
        # make chunk
        chunk_words = words[start:end]
        # join words back into normal text
        chunk = " ".join(chunk_words)
        # add chunk
        chunks.append(chunk)
        # Move forward while keeping some overlap.
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Convert complete documents into smaller chunks
    while preserving source metadata.
    """

    all_chunks = []

    for document in documents:
        # split document into smaller chunks
        text_chunks = chunk_text(document["text"])

        # give every chunk metadata
        for chunk_number, text in enumerate(text_chunks):

            chunk = {
                "chunk_id": f"{document['source']}-{chunk_number}",
                "source": document["source"],
                "chunk_number": chunk_number,
                "text": text,}

            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":

    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.\n")

    for chunk in chunks:
        print("=" * 70)
        print(f"CHUNK ID: {chunk['chunk_id']}")
        print(f"SOURCE: {chunk['source']}")
        print(f"CHUNK NUMBER: {chunk['chunk_number']}")
        print("-" * 70)
        print(chunk["text"])
        print()