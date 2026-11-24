
import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from text_splitter import split_documents
from document_loader import load_documents


load_dotenv()

MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768


def create_embedding_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    return genai.Client(api_key=api_key)


def normalize_embedding(values):
    magnitude = math.sqrt(sum(value * value for value in values))

    if magnitude == 0:
        raise ValueError("Received a zero-magnitude embedding")

    return [value / magnitude for value in values]


def embed_documents(chunks):
    if not chunks:
        raise ValueError("No chunks received for embedding")

    client = create_embedding_client()

    texts = [chunk["content"] for chunk in chunks]

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=OUTPUT_DIMENSIONALITY
        )
    )

    if not response.embeddings:
        raise ValueError("No embeddings returned by Google API")

    if len(response.embeddings) != len(chunks):
        raise ValueError("Number of embeddings does not match number of chunks")

    embedded_documents = []

    for chunk, embedding in zip(chunks, response.embeddings):
        normalized_values = normalize_embedding(embedding.values)

        embedded_documents.append({
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "embedding": normalized_values
        })

    return embedded_documents


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    embedded_documents = embed_documents(chunks)

    print("Embedding generation successful!")
    print("Model:", MODEL_NAME)
    print("Total chunks:", len(embedded_documents))
    print(
        "Embedding dimensions:",
        len(embedded_documents[0]["embedding"])
    )

    first_embedding = embedded_documents[0]["embedding"]

    norm = math.sqrt(sum(value * value for value in first_embedding))

    print("First embedding magnitude:", round(norm, 6))
    print("First chunk source:", embedded_documents[0]["metadata"]["source"])
    print("First chunk preview:", embedded_documents[0]["content"][:150])
