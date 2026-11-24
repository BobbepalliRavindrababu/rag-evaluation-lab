
from pathlib import Path

import chromadb
from google.genai import types

from embeddings import create_embedding_client, normalize_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "nexora_policies"
MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768


def create_query_embedding(question):
    client = create_embedding_client()

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=OUTPUT_DIMENSIONALITY
        )
    )

    if not response.embeddings:
        raise ValueError("Query embedding generation failed")

    return normalize_embedding(response.embeddings[0].values)


def retrieve_documents(question, top_k=3):
    if not question.strip():
        raise ValueError("Question cannot be empty")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise ValueError("Vector collection is empty")

    query_embedding = create_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for index, content in enumerate(results["documents"][0]):
        retrieved_documents.append({
            "content": content,
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index]
        })

    return retrieved_documents
