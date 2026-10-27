import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from src.config import settings


def get_chroma_client(path: str = "./chroma_store"):
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(
        path=path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_vector_store(client, embedding_function, collection_name: str | None = None):
    collection_name = collection_name or settings.chroma_collection
    return Chroma(
        client=client,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )


def load_to_vector_store(client, embedding, documents, collection_name: str):
    return Chroma.from_documents(
        client=client,
        embedding=embedding,
        documents=documents,
        collection_name=collection_name,
    )

     
