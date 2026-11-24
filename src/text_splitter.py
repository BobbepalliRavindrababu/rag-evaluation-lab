
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    split_documents = []

    for document in documents:

        chunks = splitter.split_text(document["content"])

        for index, chunk in enumerate(chunks):

            split_documents.append({
                "content": chunk,
                "metadata": {
                    "source": document["metadata"]["source"],
                    "chunk_id": index
                }
            })

    return split_documents


if __name__ == "__main__":

    documents = load_documents()

    chunks = split_documents(documents)

    print("Total documents:", len(documents))
    print("Total chunks:", len(chunks))

    for chunk in chunks[:5]:
        print("\n" + "=" * 50)
        print("Source:", chunk["metadata"]["source"])
        print("Chunk ID:", chunk["metadata"]["chunk_id"])
        print("Content:")
        print(chunk["content"])
