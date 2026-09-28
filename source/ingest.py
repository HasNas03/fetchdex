from pathlib import Path

# Path to the folder containing our knowledge documents.
DOCUMENTS_FOLDER = Path("documents")

def load_documents() -> list[dict]:
    """
    Load every .txt file from the documents folder
    Return a list of dictionaries, e.g.:
        [
            {
                "source": "abc.txt",
                "text": "..."
            },
            {
                "source": "xyz.txt",
                "text": "..."
            }
        ]
    """

    # This will hold every document we load.
    documents = []
    # find every .txt
    for file_path in DOCUMENTS_FOLDER.glob("*.txt"):
        # read the entire file
        text = file_path.read_text(encoding="utf-8")
        # store name and contents
        document = {
            "source": file_path.name,
            "text": text,
        }
        # Add this document to our list.
        documents.append(document)

    return documents


if __name__ == "__main__":
    # run function to load all documents from /documents
    documents = load_documents()
    # show how many were found.
    print(f"Loaded {len(documents)} documents for context.\n")
    # print each document name
    for document in documents:
        print("=" * 60)
        print(f"SOURCE: {document['source']}")