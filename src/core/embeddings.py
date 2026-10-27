import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings


class OllamaEmbeddingFunction:
    """Local Ollama embedder via the /api/embed HTTP endpoint.

    nomic-style models are train-time prefixed: docs get 'search_document: ',
    queries get 'search_query: '. Skipping the prefix silently degrades
    retrieval quality, so it is applied here automatically.
    """

    def __init__(self, model: str = "nomic-embed-text",
                 url: str = "http://localhost:11434/api/embed"):
        self.model = model
        self.url = url

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        resp = requests.post(self.url, timeout=120, json={
            "model": self.model,
            "input": [prefix + t for t in texts],
        })
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts):
        return self._embed(texts, "search_document: ")

    def embed_query(self, text):
        return self._embed([text], "search_query: ")[0]

    def __call__(self, input):
        return self.embed_documents(input)


class LocalEmbeddingFunction:
    """Opt-in local embedder; lazy import so nothing downloads if unused."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return self._ef(texts)

    def embed_query(self, text):
        return self._ef([text])[0]

    def __call__(self, input):
        return self._ef(input)


def get_embedding_model(use_local: bool = False, embedder: str | None = None):
    """Google Gemini embeddings by default; MiniLM/Ollama are opt-in only."""
    if embedder == "ollama":
        return OllamaEmbeddingFunction()
    if embedder == "gemini":
        pass  # fall through to the API default
    elif use_local:
        return LocalEmbeddingFunction()
    return GoogleGenerativeAIEmbeddings(
        api_key=settings.google_api_key,
        model="gemini-embedding-2-preview",
    )



