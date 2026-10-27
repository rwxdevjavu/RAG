from src.core.vector_store import get_chroma_client, get_vector_store
from src.core.embeddings import get_embedding_model


def get_retriever(k: int = 4, collection_name: str | None = None):
    client = get_chroma_client()
    embeddings = get_embedding_model()
    vectorstore = get_vector_store(
        client=client,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

