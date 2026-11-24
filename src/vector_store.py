
from pathlib import Path

import chromadb

from document_loader import load_documents
from text_splitter import split_documents
from embeddings import embed_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "nexora_policies"


def create_vector_store():
    # Create a persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # Create or retrieve our collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Nexora Technologies policy documents"}
    )

    # Load, split and embed documents
    documents = load_documents()
    chunks = split_documents(documents)
    embedded_documents = embed_documents(chunks)

    # Prepare data for ChromaDB
    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for item in embedded_documents:
        source = item["metadata"]["source"]
        chunk_id = item["metadata"]["chunk_id"]

        ids.append(f"{source}_{chunk_id}")
        texts.append(item["content"])
        metadatas.append(item["metadata"])
        embeddings.append(item["embedding"])

    # Upsert inserts new records or updates existing records
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return collection


if __name__ == "__main__":
    collection = create_vector_store()

    print("Vector storage successful!")
    print("Collection:", collection.name)
    print("Total stored chunks:", collection.count())
    print("Database location:", CHROMA_PATH)
