"""Retriever eval runner — query the local Chroma collection, score with eval_metrics.

Pipeline per query:
    claim text -> Gemini embedding -> Chroma top-k chunks
    -> collapse chunks to doc ids (first occurrence = best rank)
    -> compare against qrels via bench/eval_metrics.py -> mean over queries

Usage:
    .venv/bin/python bench/run_eval.py --collection scifact_v1 \
        --queries dataset/scifact/queries.jsonl \
        --qrels dataset/scifact/qrels/test.tsv --k 1,5,10

    add --limit 5 for a cheap smoke run, --limit 0 to just check dataset plumbing
    (writes per-query JSON to --out for the error-analysis loop)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.eval_metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_chroma_client

# Chroma ranks CHUNKS; BEIR grades DOCUMENTS. To grade an honest top-k DOCS we
# must fetch k * headroom chunks (a doc can spend several slots on its chunks)
# and collapse to doc ids first. 5x covers corpora whose docs chunk up to ~5.
DOC_WINDOW_HEADROOM = 5


def load_queries(path: Path) -> list[dict]:
    """BEIR queries.jsonl: one {"_id": ..., "text": ...} per line."""
    return [
        {"id": str(d["_id"]), "text": d["text"]}
        for line in open(path, encoding="utf-8")
        if (d := json.loads(line))
    ]


def load_qrels(path: Path) -> dict[str, set[str]]:
    """TREC qrels TSV: query-id \t corpus-id \t score; score > 0 = relevant."""
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 3 or row[0] == "query-id":
                continue
            qid, doc_id, score = row
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def collapse_chunks_to_docs(chunk_doc_ids: list[str]) -> list[str]:
    """Ordered dedupe: a doc's first (best-ranked) chunk position is its rank."""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in chunk_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score chroma retrieval against BEIR qrels")
    parser.add_argument("--collection", required=True, help="chroma collection to query")
    parser.add_argument("--queries", type=Path, default=Path("dataset/scifact/queries.jsonl"))
    parser.add_argument("--qrels", type=Path, default=Path("dataset/scifact/qrels/test.tsv"))
    parser.add_argument("--k", default="1,5,10", help="comma-separated k values, e.g. 1,5,10")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "bench" / "eval_results.json")
    parser.add_argument("--limit", type=int, default=None, help="eval only the first N queries (smoke)")
    parser.add_argument("--use-local", action="store_true", help="opt-in local MiniLM embedder")
    parser.add_argument("--embedder", choices=["gemini", "local", "ollama"], default=None,
                        help="explicit embedder choice (overrides --use-local)")
    parser.add_argument(
        "--corpus", type=Path, default=Path("dataset/scifact/corpus.jsonl"),
        help="corpus.jsonl used to verify every retrieved id is a real document id",
    )
    parser.add_argument("--force", action="store_true", help="overwrite the --out file if it exists")
    args = parser.parse_args()

    if args.limit != 0 and args.out.exists() and not args.force:
        print(f"refusing to overwrite existing results {args.out} — pass --force")
        return

    ks = sorted({int(x) for x in args.k.split(",")})
    k_max = max(ks)

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)

    covered = [q for q in queries if q["id"] in qrels]
    if args.limit == 0:
        print(f"plumbing check: {len(queries)} queries | {len(qrels)} qrels keys | "
              f"queries with gold: {len(covered)} | k: {ks}")
        return
    if args.limit is not None:
        queries = queries[: args.limit]

    # ground-truth contract: every graded id must be a known document id
    corpus_ids: set[str] | None = None
    if args.corpus.exists():
        corpus_ids = {str(json.loads(line)["_id"]) for line in open(args.corpus)}
        print(f"ID contract: {len(corpus_ids)} corpus doc-ids loaded from {args.corpus}")

    no_gold = [q["id"] for q in queries if q["id"] not in qrels]
    if no_gold:
        print(f"warning: {len(no_gold)} queries have no qrels; they will score 0 everywhere")

    client = get_chroma_client(path="./chroma_store")
    collection = client.get_collection(args.collection)
    embedder = get_embedding_model(use_local=args.use_local,
                                   embedder=args.embedder or ("local" if args.use_local else None))
    embedder_name = getattr(embedder, "model", "local-default")
    print(f"collection {args.collection!r}: {collection.count()} chunks | "
          f"queries: {len(queries)} | embedder: {type(embedder).__name__}")

    missing_metadata = 0
    per_query: list[dict] = []
    latency_ms: list[float] = []
    accs = {m: {k: [] for k in ks} for m in ("recall", "precision", "mrr", "ndcg")}

    for i, q in enumerate(queries, start=1):
        expected = qrels.get(q["id"], set())

        t0 = time.perf_counter()
        query_vec = embedder.embed_query(q["text"])
        res = collection.query(
            query_embeddings=[query_vec],
            n_results=k_max * DOC_WINDOW_HEADROOM,  # chunks fetched; graded set is DOCS
            include=["metadatas", "distances"],
        )
        latency_ms.append((time.perf_counter() - t0) * 1000)

        metas = res["metadatas"][0]
        chunk_doc_ids = []
        for m in metas:
            if "doc_id" in m:
                chunk_doc_ids.append(m["doc_id"])
            else:
                missing_metadata += 1
        retrieved = collapse_chunks_to_docs(chunk_doc_ids)

        if len(retrieved) < k_max:
            print(f"warning: query {q['id']} yielded only {len(retrieved)} distinct docs for k={k_max} "
                  f"({len(chunk_doc_ids)} chunks fetched); increase DOC_WINDOW_HEADROOM")
        if corpus_ids is not None and not set(retrieved) <= corpus_ids:
            unknown = sorted(set(retrieved) - corpus_ids)[:3]
            raise RuntimeError(
                f"ID contract violated for query {q['id']}: retrieved ids not in corpus.jsonl "
                f"(e.g. {unknown}). The collection's doc_id metadata does not match the qrels' id space."
            )

        row = {"id": q["id"], "text": q["text"], "gold": sorted(expected),
               "retrieved_docs": retrieved[:k_max]}
        for k in ks:
            for name, fn in (
                ("recall", recall_at_k),
                ("precision", precision_at_k),
                ("mrr", mrr_at_k),
                ("ndcg", ndcg_at_k),
            ):
                score = fn(retrieved, expected, k)
                accs[name][k].append(score)
                row[f"{name}@{k}"] = round(score, 4)
        per_query.append(row)

        worst = f"recall@{ks[0]}={accs['recall'][ks[0]][-1]:.2f}"
        print(f"[{i:>3}/{len(queries)}] {q['id']}  {worst}", flush=True)

    if missing_metadata:
        print(f"warning: {missing_metadata} chunks had no doc_id metadata — "
              f"were they ingested with the BEIR/legacy doc_id logic?")

    print("\n=== RESULTS (mean over all queries) ===")
    summary = {"embedder": embedder_name, "collection": args.collection,
               "k": ks, "num_queries": len(queries)}
    for k in ks:
        r = sum(accs["recall"][k]) / len(queries)
        p = sum(accs["precision"][k]) / len(queries)
        m = sum(accs["mrr"][k]) / len(queries)
        n = sum(accs["ndcg"][k]) / len(queries)
        print(f"k={k:<3} recall={r:.3f} precision={p:.3f} mrr={m:.3f} ndcg={n:.3f}")
        summary.update({f"recall@{k}": r, f"precision@{k}": p,
                        f"mrr@{k}": m, f"ndcg@{k}": n})
    summary.update({
        "latency_ms_mean": round(sum(latency_ms) / len(latency_ms), 1),
        "latency_ms_p95": round(percentile(latency_ms, 0.95), 1),
    })
    print(f"latency mean {summary['latency_ms_mean']} ms  p95 {summary['latency_ms_p95']} ms")

    args.out.write_text(json.dumps(
        {"summary": summary, "per_query": per_query}, indent=2
    ))
    print(f"\nper-query rows + summary written to {args.out}")


if __name__ == "__main__":
    main()
