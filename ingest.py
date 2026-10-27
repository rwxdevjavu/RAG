import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_chroma_client, load_to_vector_store

BATCH_SIZE = 100


def _split(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into a local Chroma collection")
    parser.add_argument(
        "--docs",
        required=True,
        help="either a directory of .txt files (doc_id = part before '__') or a BEIR-style .jsonl corpus (_id field)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: settings.chroma_collection)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--use-local", action="store_true",
                        help="embed with the local MiniLM model instead of the Google API")
    parser.add_argument("--embedder", choices=["gemini", "local", "ollama"], default=None,
                        help="explicit embedder choice (overrides --use-local)")
    args = parser.parse_args()
    collection_name = args.collection or settings.chroma_collection
    embedder_choice = args.embedder or ("local" if args.use_local else None)

    client = get_chroma_client(path="./chroma_store")
    embedder = get_embedding_model(use_local=args.use_local, embedder=embedder_choice)

    # Reset the collection so reruns start clean (chunk sets are never duplicated;
    # rerunning intentionally re-embeds because the collection is wiped first).
    try:
        client.delete_collection(collection_name)
        print(f"reset existing collection {collection_name!r}")
    except Exception:
        pass

    source = Path(args.docs)
    split_docs: list[Document] = []

    if source.suffix == ".jsonl":
        # BEIR-style corpus: one JSON doc per line with _id / title / text.
        with open(source, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                doc_id = str(d["_id"])
                text = (d.get("title", "") + "\n\n" + d.get("text", "")).strip()
                split_docs.extend(
                    _split(
                        [Document(page_content=text, metadata={"source": source.name, "doc_id": doc_id})],
                        args.chunk_size,
                        args.chunk_overlap,
                    )
                )
    else:
        for file in sorted(p for p in source.iterdir() if p.is_file() and p.suffix == ".txt"):
            doc_id = file.name.split("__")[0]
            split_docs.extend(
                _split(
                    [
                        Document(
                            page_content=file.read_text(encoding="utf-8"),
                            metadata={"source": str(file.resolve()), "doc_id": doc_id},
                        )
                    ],
                    args.chunk_size,
                    args.chunk_overlap,
                )
            )

    for i in range(0, len(split_docs), BATCH_SIZE):
        load_to_vector_store(
            documents=split_docs[i : i + BATCH_SIZE],
            client=client,
            embedding=embedder,
            collection_name=collection_name,
        )
        print(f"ingested {min(i + BATCH_SIZE, len(split_docs))}/{len(split_docs)}")

    col = client.get_collection(collection_name)
    print(
        f"ingest completed: {col.count()} chunks from {len(split_docs)} docs "
        f"into {collection_name!r} (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})"
    )


if __name__ == "__main__":
    main()
