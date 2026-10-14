"""Retrieval evaluation: Recall@k and MRR over ``samples/queries.json``.

    python -m pagex.eval.retrieval            # needs an indexed corpus
    python -m pagex.eval.retrieval --json

Gate (build manual WP7): **Recall@5 >= 0.8** on the sample query set.
Each query in queries.json carries ``expected_doc`` (a path fragment of the
sample that should win) and optionally ``expected_page``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagex.core.retrieve import recall_at_k, reciprocal_rank, search  # noqa: E402
from pagex.core.store import Store                                    # noqa: E402

QUERIES_PATH = ROOT / "samples" / "queries.json"


def load_queries(path: str | Path | None = None) -> list[dict]:
    p = Path(path or QUERIES_PATH)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [q for q in data if q.get("query")]


def resolve_expected(store: Store, expected: str) -> list[str]:
    """Map a sample path fragment (``notes/1_x.png``) to indexed doc ids."""
    stem = Path(expected).stem
    rows = store.conn.execute(
        "SELECT doc_id, source_path, title FROM documents").fetchall()
    hits = []
    for r in rows:
        if expected and expected in (r["source_path"] or ""):
            hits.append(r["doc_id"])
        elif stem and stem.lower() in (r["source_path"] or "").lower():
            hits.append(r["doc_id"])
        elif stem and stem.lower() in (r["title"] or "").lower():
            hits.append(r["doc_id"])
    return sorted(set(hits))


def evaluate(store: Store, queries: Sequence[dict] | None = None, *,
             k: int = 5, embedder=None, use_graph: bool = True) -> dict:
    queries = list(queries if queries is not None else load_queries())
    per_query = []
    latencies: list[float] = []
    hits = 0
    mrr_total = 0.0
    page_hits = 0
    skipped = 0

    for q in queries:
        expected_ids = resolve_expected(store, q.get("expected_doc", ""))
        if not expected_ids:
            skipped += 1
            per_query.append({"query": q["query"], "expected": q.get("expected_doc"),
                              "resolved": None, "hit": None})
            continue
        t0 = time.perf_counter()
        kwargs = {"embedder": embedder} if embedder else {}
        result = search(store, q["query"], k=k, use_graph=use_graph, **kwargs)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranked_docs = [c.doc_id for c in result.citations]
        hit_docs = [d for d in ranked_docs[:k] if d in expected_ids]
        hit = bool(hit_docs)
        hits += 1 if hit else 0
        rr = 0.0
        for i, d in enumerate(ranked_docs, start=1):
            if d in expected_ids:
                rr = 1.0 / i
                break
        mrr_total += rr

        exp_page = q.get("expected_page")
        page_ok = None
        if hit and exp_page:
            page_ok = any(c.page_no == exp_page
                          for c in result.citations[:k] if c.doc_id in expected_ids)
            page_hits += 1 if page_ok else 0
        per_query.append({
            "query": q["query"], "expected": q.get("expected_doc"),
            "resolved": expected_ids, "hit": hit, "rr": round(rr, 3),
            "top": [{"doc": c.title or c.doc_id, "page": c.page_no,
                     "score": round(c.score, 4), "src": c.source}
                    for c in result.citations[:k]],
            "page_ok": page_ok,
            "ms": round(latencies[-1], 1),
        })

    n = len(per_query) - skipped
    latencies.sort()
    return {
        "queries_total": len(queries),
        "queries_scored": n,
        "queries_unresolved": skipped,
        "k": k,
        "recall_at_k": round(hits / n, 3) if n else 0.0,
        "mrr": round(mrr_total / n, 3) if n else 0.0,
        "page_accuracy": round(page_hits / n, 3) if n else None,
        "latency_ms": {"p50": round(latencies[len(latencies) // 2], 1)
                       if latencies else 0.0,
                       "p95": round(latencies[int(len(latencies) * 0.95)], 1)
                       if latencies else 0.0,
                       "max": round(latencies[-1], 1) if latencies else 0.0},
        "gate_recall_at_5": (hits / n) >= 0.8 if n else False,
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--show", action="store_true", help="per-query detail")
    args = ap.parse_args(argv)

    store = Store(args.db) if args.db else Store()
    report = evaluate(store, load_queries(args.queries) if args.queries else None,
                      k=args.k, use_graph=not args.no_graph)
    store.close_all()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["gate_recall_at_5"] else 1

    print(f"queries {report['queries_scored']}/{report['queries_total']} scored "
          f"({report['queries_unresolved']} unresolved)")
    print(f"Recall@{report['k']}: {report['recall_at_k']:.3f}   "
          f"MRR: {report['mrr']:.3f}   "
          f"page acc: {report['page_accuracy']}")
    lat = report["latency_ms"]
    print(f"latency p50 {lat['p50']} ms · p95 {lat['p95']} ms · max {lat['max']} ms")
    print("GATE:", "PASS (Recall@5 >= 0.80)" if report["gate_recall_at_5"]
          else "FAIL (Recall@5 < 0.80)")
    if args.show or not report["gate_recall_at_5"]:
        for q in report["per_query"]:
            mark = "x" if q.get("hit") is False else ("?" if q.get("hit") is None else "o")
            print(f" [{mark}] {q['query'][:60]}")
            if q.get("top"):
                for t in q["top"][:3]:
                    print(f"        {t['doc'][:40]} p{t['page']} "
                          f"{t['score']} ({t['src']})")
    return 0 if report["gate_recall_at_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
