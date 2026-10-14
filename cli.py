#!/usr/bin/env python3
"""PageX command line - a thin shell over ``pagex.core`` (same code as the GUI).

    python cli.py list
    python cli.py ingest samples/notes/1_rotational_motion.png --handwritten
    python cli.py ingest samples/digital/page_sample.pdf
    python cli.py search "angular momentum conservation"
    python cli.py ask "What is the moment of inertia?"
    python cli.py chat
    python cli.py graph
    python cli.py convert-notes notes/1.png --out x.tex --formats tex,docx,pdf
    python cli.py bench
    python cli.py watch <dir>          # one-shot scan of a watch folder

Everything is offline: no cloud API, no telemetry.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagex.core.config import CFG, setup_logging            # noqa: E402
from pagex.core.store import Store                          # noqa: E402

log = logging.getLogger("pagex")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _store(args) -> Store:
    return Store(args.db) if getattr(args, "db", None) else Store()


def _print_citations(citations, limit: int = 5) -> None:
    for i, c in enumerate(citations[:limit], start=1):
        src = c.source
        print(f"  {i}. {c.title} · page {c.page_no} · {src} · "
              f"score {c.score:.4f} · {c.chunk_id}")
        if c.snippet:
            print(f"     {c.snippet[:160]}")


def _resolve_docs(store: Store, names: list[str] | None) -> list[str] | None:
    """Accept doc ids or title fragments for --docs."""
    if not names:
        return None
    docs = store.list_documents()
    ids = []
    for name in names:
        if name in {d["doc_id"] for d in docs}:
            ids.append(name)
            continue
        matches = [d for d in docs if name.lower() in d["title"].lower()]
        ids.extend(m["doc_id"] for m in matches)
    return sorted(set(ids)) or None


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_list(args) -> int:
    store = _store(args)
    docs = store.list_documents()
    if not docs:
        print("No documents indexed yet. Try:")
        print("  python cli.py ingest samples/notes/1_rotational_motion.png --handwritten")
        return 0
    if args.json:
        print(json.dumps(docs, indent=2, default=str))
        return 0
    print(f"{'ID':<20} {'STATUS':<9} {'PAGES':>5} {'CHUNKS':>7}  TITLE")
    for d in docs:
        print(f"{d['doc_id']:<20} {d['status']:<9} {d['n_pages']:>5} "
              f"{d.get('n_chunks', 0):>7}  {d['title']}")
    stats = store.kg_stats()
    print(f"\n{len(docs)} document(s) · {store.count_chunks()} chunks · "
          f"KG {stats['nodes']} nodes / {stats['edges']} edges · {store.db_path}")
    return 0


def cmd_ingest(args) -> int:
    from pagex.core.pipeline import ingest
    store = _store(args)
    paths = []
    for raw in args.path:
        p = Path(raw)
        if p.is_dir():
            from pagex.core.watch import scan
            paths.extend(scan(p))
        else:
            paths.append(p)
    if not paths:
        print("nothing to ingest (no images/PDFs found)", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    failures = 0
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name} ...")

        def progress(stage: str, cur: int, total: int) -> None:
            print(f"    {stage:<11} {cur}/{total}")

        try:
            report = ingest(path, title=args.title, handwritten=args.handwritten,
                            store=store, progress=progress,
                            use_kg=not args.no_kg)
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        print(f"    -> {report.doc_id}: {report.n_pages} pages, "
              f"{report.n_blocks} blocks, {report.n_chunks} chunks, "
              f"KG {report.n_kg_nodes}/{report.n_kg_edges} in {report.seconds:.1f}s")
        if report.tex_path:
            print(f"    -> {report.tex_path}")
        for err in report.errors:
            print(f"    warning: {err}", file=sys.stderr)
        if not report.ok:
            failures += 1
    store.close_all()
    print(f"done in {time.perf_counter() - t0:.1f}s")
    return 1 if failures else 0


def cmd_convert(args) -> int:
    from pagex.core.pipeline import convert_to_latex
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dir = Path(args.out).parent if args.out and args.out != "." else None
    base = Path(args.out).stem if args.out and args.out != "." else None
    written = convert_to_latex(args.path, title=args.title,
                               handwritten=not args.digital,
                               out_formats=formats, out_dir=out_dir,
                               base_name=base)
    for fmt, path in written.items():
        print(f"{fmt:<10} -> {path} ({path.stat().st_size} bytes)")
    return 0


def cmd_search(args) -> int:
    from pagex.core.retrieve import search
    store = _store(args)
    t0 = time.perf_counter()
    result = search(store, args.query, k=args.top_k,
                    doc_ids=_resolve_docs(store, args.docs),
                    use_graph=not args.no_graph)
    dt = (time.perf_counter() - t0) * 1000
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if not result.citations:
        print("no results")
        return 1
    print(f"'{args.query}' -> {len(result.citations)} result(s) in {dt:.0f} ms "
          f"(vector {len(result.rankers.get('vector', []))}, "
          f"bm25 {len(result.rankers.get('bm25', []))}, "
          f"graph {len(result.rankers.get('graph', []))})")
    _print_citations(result.citations, args.top_k)
    return 0


def cmd_ask(args) -> int:
    from pagex.core.rag import answer_query
    store = _store(args)
    result = answer_query(store, args.query, k=args.top_k,
                          doc_ids=_resolve_docs(store, args.docs),
                          session_id=args.session)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(result.answer)
    if result.citations:
        print("\nsources:")
        for c in result.citations:
            print(f"  [doc:{c.title} p.{c.page_no}]  {c.snippet[:120]}")
    if result.gen:
        print(f"\n({result.gen.tokens} tokens, {result.gen.tok_per_s:.1f} tok/s, "
              f"{result.seconds:.1f}s total, rss {result.gen.peak_rss_gb:.2f} GB)")
    return 0


def cmd_chat(args) -> int:
    from pagex.core.rag import chat
    store = _store(args)
    chat(store, session_id=args.session, k=args.top_k)
    store.close_all()
    return 0


def cmd_graph(args) -> int:
    store = _store(args)
    stats = store.kg_stats()
    if args.json:
        nodes, edges = store.kg_subgraph([])
        print(json.dumps({"stats": stats}, indent=2))
        return 0
    print(f"knowledge graph: {stats['nodes']} nodes, {stats['edges']} edges")
    rows = store.conn.execute(
        """SELECT rel, COUNT(*) n FROM kg_edges GROUP BY rel ORDER BY n DESC"""
    ).fetchall()
    for r in rows:
        print(f"  {r['rel']:<12} {r['n']}")
    print("\ntop concepts:")
    for r in store.conn.execute(
            """SELECT label, kind FROM kg_nodes ORDER BY kind, label LIMIT 25"""):
        print(f"  [{r['kind']:<9}] {r['label']}")
    if args.dot:
        _print_dot(store)
    return 0


def _print_dot(store: Store) -> None:
    print("\n--- graphviz ---\n digraph pagex {")
    for n in store.conn.execute("SELECT node_id,kind,label FROM kg_nodes"):
        print(f'  "{n["node_id"]}" [label="{_esc(n["label"])}" '
              f'shape={("box" if n["kind"] == "document" else "ellipse")}];')
    for e in store.conn.execute("SELECT src,dst,rel FROM kg_edges"):
        print(f'  "{e["src"]}" -> "{e["dst"]}" [label="{e["rel"]}"];')
    print("}")


def _esc(text: str) -> str:
    return (text or "").replace("\\", "/").replace('"', "'")[:60]


def cmd_bench(args) -> int:
    import tools.probe as probe      # noqa: F401  (path already set)
    from tools import probe as _probe
    return _probe.main([] if not args.fast else ["--fast"])


def cmd_watch(args) -> int:
    from pagex.core.pipeline import ingest
    from pagex.core.watch import WatchState, scan
    store = _store(args)
    target = Path(args.dir)
    files = scan(target)
    if not files:
        print(f"no ingestible files in {target}")
        return 0
    state = WatchState(target, debounce_s=0.0)
    state.on_change(None)
    ready = state.drain_now() or files
    for path in ready:
        print(f"ingesting {path.name} ...")
        try:
            report = ingest(path, store=store, handwritten=args.handwritten)
            print(f"  -> {report.n_blocks} blocks, {report.n_chunks} chunks")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
    store.close_all()
    return 0


def cmd_smoke(args) -> int:
    from pagex.testing.smoke import main as smoke_main
    return smoke_main([])


def cmd_gui(args) -> int:
    from pagex.gui.app import main as gui_main
    argv: list[str] = []
    if args.db:
        argv += ["--db", args.db]
    if args.ingest:
        argv += ["--ingest", *args.ingest]
    if args.verbose:
        argv.append("--verbose")
    return gui_main(argv)


def cmd_models(args) -> int:
    from pagex.core.models import model_report
    rep = model_report()
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"tier {rep['tier']} · threads {rep['threads']} · rss {rep['rss_gb']} GB")
    for key, info in rep["files"].items():
        mark = "ok " if info["exists"] else "MISS"
        print(f"  [{mark}] {key:<18} {info['mb']:>8} MB  {info['path']}")
    print(f"heavy slot: {rep['heavy']['state']}")
    if CFG.missing_models():
        print("\nmissing models -> python tools/download_models.py")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pagex", description="PageX - offline document intelligence")
    ap.add_argument("--db", default=None, help="override the SQLite path")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("list", help="list indexed documents")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("ingest", help="index an image / PDF / folder")
    p.add_argument("path", nargs="+")
    p.add_argument("--title", default=None)
    p.add_argument("--handwritten", action="store_true",
                   help="treat the input as handwritten notes")
    p.add_argument("--no-kg", action="store_true", help="skip knowledge graph")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("convert-notes", help="image/PDF -> LaTeX (+ exports)")
    p.add_argument("path")
    p.add_argument("--out", default=None, help="output file stem/path")
    p.add_argument("--title", default=None)
    p.add_argument("--formats", default="tex",
                   help="comma list: tex,clean-tex,pdf,docx,html,json,md")
    p.add_argument("--digital", action="store_true",
                   help="input is born-digital, not handwriting")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("search", help="hybrid search (vector+BM25+graph)")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=CFG.top_k)
    p.add_argument("--docs", nargs="*", default=None)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("ask", help="grounded RAG answer with citations")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=CFG.top_k)
    p.add_argument("--docs", nargs="*", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("chat", help="interactive multi-turn chat")
    p.add_argument("--session", default=None)
    p.add_argument("--top-k", type=int, default=CFG.top_k)
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("graph", help="inspect the knowledge graph")
    p.add_argument("--json", action="store_true")
    p.add_argument("--dot", action="store_true", help="print graphviz DOT")
    p.set_defaults(fn=cmd_graph)

    p = sub.add_parser("bench", help="run tools/probe.py")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("models", help="show model files / slot state")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("watch", help="ingest every file in a folder")
    p.add_argument("dir")
    p.add_argument("--handwritten", action="store_true")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("smoke", help="pre-demo self check (no heavy models)")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("gui", help="launch the desktop shell (PyQt6)")
    p.add_argument("--db", default=None, help="override SQLite path")
    p.add_argument("--ingest", nargs="*", default=None,
                   help="ingest these files/folders on startup")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_gui)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:                          # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
