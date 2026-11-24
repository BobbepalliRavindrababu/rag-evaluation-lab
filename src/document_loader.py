
from pathlib import Path


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Location of our source documents
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents"


def load_documents():
    documents = []

    # Find all Markdown files
    files = list(DOCUMENTS_PATH.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"No Markdown documents found in: {DOCUMENTS_PATH}"
        )

    for file_path in files:

        # Read the document content
        content = file_path.read_text(encoding="utf-8")

        # Store content along with its metadata
        document = {
            "content": content,
            "metadata": {
                "source": file_path.name
            }
        }

        documents.append(document)

    return documents


if __name__ == "__main__":
    loaded_documents = load_documents()

    print(f"Total documents loaded: {len(loaded_documents)}")

    for document in loaded_documents:
        print("\n" + "=" * 50)
        print("Source:", document["metadata"]["source"])
        print("Characters:", len(document["content"]))
        print("Preview:")
        print(document["content"][:200])
