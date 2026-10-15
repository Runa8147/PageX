#!/usr/bin/env python3
"""PageX command line - a thin shell over ``pagex.core`` (same code as the GUI).

    pagex list
    pagex ingest samples/notes/1_rotational_motion.png --handwritten
    pagex ingest "Artificial Intelligence handwitten notes.pdf" --handwritten
    pagex search "angular momentum conservation"
    pagex ask "What is the moment of inertia?"
    pagex chat
    pagex graph
    pagex convert-notes notes/1.png --out x.tex --formats tex,docx,pdf
    pagex bench                    # hardware probe -> bench.json
    pagex models list|download|verify|install-pack
    pagex demo                     # seed the sample corpus, print queries
    pagex doctor                   # paths / deps / model status (bug reports)
    pagex watch <dir>              # one-shot scan of a watch folder

Installed entry point (``pip install`` / frozen build) is ``pagex``; the
repository-root ``cli.py`` is a shim over this module so there is exactly one
implementation.  Everything is offline: no cloud API, no telemetry.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from .core import paths
from .core.config import CFG, load_config, setup_logging
from .core.store import Store

log = logging.getLogger("pagex")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _configure_console() -> None:
    """UTF-8 console output on Windows (cp1252 would crash on ``·``/``→``)."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is None:
            continue
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:                       # pragma: no cover
            pass


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
        print("  pagex demo                      # seed the sample corpus")
        print("  pagex ingest <file.png|file.pdf> --handwritten")
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
    from .core.pipeline import ingest
    store = _store(args)
    paths_in = []
    for raw in args.path:
        p = Path(raw)
        if p.is_dir():
            from .core.watch import scan
            paths_in.extend(scan(p))
        else:
            paths_in.append(p)
    if not paths_in:
        print("nothing to ingest (no images/PDFs found)", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    failures = 0
    for i, path in enumerate(paths_in, 1):
        print(f"[{i}/{len(paths_in)}] {path.name} ...")

        def progress(stage: str, cur: int, total: int) -> None:
            print(f"    {stage:<11} {cur}/{total}")

        try:
            report = ingest(path, title=args.title, handwritten=args.handwritten,
                            store=store, progress=progress,
                            use_kg=not args.no_kg,
                            max_pages=getattr(args, "max_pages", None))
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
    from .core.pipeline import convert_to_latex
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dir = Path(args.out).parent if args.out and args.out != "." else None
    base = Path(args.out).stem if args.out and args.out != "." else None
    written = convert_to_latex(args.path, title=args.title,
                               handwritten=not args.digital,
                               out_formats=formats, out_dir=out_dir,
                               base_name=base,
                               max_pages=getattr(args, "max_pages", None))
    for fmt, path in written.items():
        print(f"{fmt:<10} -> {path} ({path.stat().st_size} bytes)")
    return 0


def cmd_search(args) -> int:
    from .core.retrieve import search
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
    from .core.rag import answer_query
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
    from .core.rag import chat
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
    from .probe import main as probe_main
    return probe_main([] if not args.fast else ["--fast"])


def cmd_watch(args) -> int:
    from .core.pipeline import ingest
    from .core.watch import WatchState, scan
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
    from .testing.smoke import main as smoke_main
    return smoke_main([])


def cmd_gui(args) -> int:
    from .gui.app import main as gui_main
    argv: list[str] = []
    if args.db:
        argv += ["--db", args.db]
    if args.ingest:
        argv += ["--ingest", *args.ingest]
    if args.verbose:
        argv.append("--verbose")
    return gui_main(argv)


def cmd_doctor(args) -> int:
    """One screen of environment facts (attach this to bug reports)."""
    from .core.model_manifest import load_manifest, missing_required, verify

    info = dict(paths.describe())
    info.update({
        "config_exists": str(CFG.path.exists()),
        "config": str(CFG.path),
        "base_dir": str(CFG.base),
        "db": str(CFG.db_path),
        "artifacts": str(CFG.artifacts_dir),
        "models_dir": str(CFG.models_dir),
        "bench": str(CFG.bench_path) + ("" if CFG.bench_path.exists()
                                        else " (missing - run `pagex bench`)"),
        "tier": CFG.effective_tier,
        "threads": CFG.effective_threads,
        "pdflatex": "yes" if _which("pdflatex") else
                    "no (ReportLab PDF fallback will be used)",
        "missing_required_models": ", ".join(missing_required()) or "none",
    })
    deps = {}
    for mod in ("numpy", "cv2", "fitz", "PIL", "PyQt6", "docx",
                "reportlab", "llama_cpp", "huggingface_hub"):
        try:
            __import__(mod)
            deps[mod] = "ok"
        except Exception as exc:
            deps[mod] = f"missing ({exc.__class__.__name__})"
    if args.json:
        print(json.dumps({"paths": info, "deps": deps,
                          "manifest": [e.key for e in load_manifest()]},
                         indent=2))
        return 0
    print("PageX doctor " + "-" * 52)
    for k in ("mode", "platform", "python", "platformdirs", "config",
              "base_dir", "db", "artifacts", "models_dir", "bench", "tier",
              "threads", "pdflatex", "missing_required_models",
              "config_exists"):
        print(f"  {k:<26} {info.get(k)}")
    print("  dependencies")
    for mod, state in deps.items():
        mark = "ok " if state == "ok" else "MISS"
        print(f"    [{mark}] {mod:<18} {state}")
    if info["missing_required_models"] != "none":
        print("\n  next: pagex models download")
    return 0


def _which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


# ---------------------------------------------------------------------------
# `pagex models` - manifest download / verify / offline pack
# ---------------------------------------------------------------------------
def cmd_models(args) -> int:
    from .core.model_manifest import (download, install_pack, load_manifest,
                                      local_path, verify)

    action = getattr(args, "action", "list") or "list"
    try:
        entries = load_manifest()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if action == "list":
        entries = _models_filter(entries, args)
        rows = []
        for e in entries:
            p = local_path(e, args.dest)
            rows.append({"key": e.key, "config_key": e.config_key,
                         "role": e.role, "required": e.required,
                         "tier": e.tier, "size_mb": round(e.size_bytes / 1e6, 1),
                         "sha256": e.sha256, "license": e.license,
                         "repo": f"{e.repo}/{e.file}", "url": e.download_url,
                         "path": str(p),
                         "exists": p.exists(),
                         "size_on_disk": p.stat().st_size if p.exists() else 0})
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        print(f"{'KEY':<14} {'TIER':>4} {'MB':>7} {'STATE':<8} ROLE")
        for r in rows:
            state = "present" if r["exists"] else "MISSING"
            print(f"{r['key']:<14} {r['tier']:>4} {r['size_mb']:>7} "
                  f"{state:<8} {r['role']}")
        have = sum(1 for r in rows if r["exists"])
        print(f"\n{have}/{len(rows)} files present in {CFG.models_dir}")
        if have < len(rows):
            print("download: pagex models download      "
                  "(verify: pagex models verify)")
        return 0

    if action == "download":
        entries = _models_filter(entries, args)
        print(f"models dir: {CFG.models_dir}")
        results = download(entries, dest_dir=args.dest)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"  [{r.get('status', '?'):<10}] {r['key']:<14} "
                      f"{r.get('detail', '')}")
        bad = [r for r in results
               if r.get("status") in ("error", "sha-mismatch")
               and (args.only or r.get("required"))]
        return 1 if bad else 0

    if action == "verify":
        entries = _models_filter(entries, args)
        rows = verify(entries, dest_dir=args.dest,
                      hash_files=not getattr(args, "no_hash", False))
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                mark = {"ok": "ok  ", "size-ok": "ok* ",
                        "no-manifest-sha": "ok* "}.get(r["status"], r["status"].upper())
                print(f"  [{mark:<12}] {r['key']:<14} {r.get('detail', '')}")
            if not args.no_hash:
                print("  (* = size checked only)")
        # missing *optional* weights are a warning, not an error
        bad = [r for r in rows
               if r["status"] in ("size-mismatch", "sha-mismatch")
               or (r["status"] == "missing" and r.get("required"))]
        return 1 if bad else 0

    if action == "install-pack":
        if not args.target:
            print("usage: pagex models install-pack <pack.zip|folder>",
                  file=sys.stderr)
            return 2
        rows = install_pack(args.target, dest_dir=args.dest)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                print(f"  [{r.get('status', '?'):<10}] {r.get('key', ''):<14} "
                      f"{r['file']} {r.get('detail', '')}")
        bad = [r for r in rows if r["status"] == "sha-mismatch"]
        return 1 if bad else 0

    if action == "pack":
        from .core.model_manifest import pack_models
        if not args.target:
            print("usage: pagex models pack <out.zip> [--only KEY ...]",
                  file=sys.stderr)
            return 2
        try:
            out = pack_models(args.dest or CFG.models_dir, args.only,
                              zip_path=args.target)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"pack written: {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return 0

    print(f"unknown models action {action!r}", file=sys.stderr)
    return 2


def _models_filter(entries, args):
    from .core.model_manifest import select
    tier = getattr(args, "tier", None)
    keys = getattr(args, "only", None)
    if keys:
        return select(entries, keys)
    if tier is not None:
        return select(entries, tier=tier)
    if getattr(args, "action", "") == "download":
        # default download: everything required plus the tier-2 fallbacks
        return select(entries, tier=2)
    return entries


# ---------------------------------------------------------------------------
# `pagex demo` - a stranger can evaluate the project in under 2 minutes
# ---------------------------------------------------------------------------
def cmd_demo(args) -> int:
    from .demo import run_demo
    return run_demo(ingest=not args.dry_run, all_samples=args.all,
                     db=args.db, json_out=args.json)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pagex", description="PageX - offline document intelligence")
    ap.add_argument("--db", default=None, help="override the SQLite path")
    ap.add_argument("--config", default=None,
                    help="use this pagex.toml instead of the platform default")
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
    p.add_argument("--max-pages", type=int, default=None,
                   help="stop after N pages (big scans: index the rest later)")
    p.add_argument("--no-kg", action="store_true", help="skip knowledge graph")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("convert-notes", help="image/PDF -> LaTeX (+ exports)")
    p.add_argument("path")
    p.add_argument("--out", default=None, help="output file stem/path")
    p.add_argument("--title", default=None)
    p.add_argument("--formats", default="tex",
                   help="comma list: tex,clean-tex,pdf,docx,html,json,md")
    p.add_argument("--max-pages", type=int, default=None,
                   help="stop after N pages")
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

    p = sub.add_parser("bench", help="hardware probe -> bench.json")
    p.add_argument("--fast", action="store_true", help="skip the VLM phase")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser(
        "models", help="list / download / verify / install GGUF weights")
    p.add_argument("action", nargs="?", default="list",
                   choices=["list", "download", "verify", "install-pack", "pack"])
    p.add_argument("target", nargs="?", default=None,
                   help="pack path (install-pack) or output zip (pack)")
    p.add_argument("--only", nargs="+", default=None,
                   help="subset of manifest keys (e.g. --only vlm llm)")
    p.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                   help="select every model up to this tier")
    p.add_argument("--dest", default=None, help="models directory override")
    p.add_argument("--no-hash", action="store_true",
                   help="verify: check sizes only, skip SHA-256")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("demo", help="seed the sample corpus and print queries")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan without ingesting")
    p.add_argument("--all", action="store_true",
                   help="ingest every sample page (slower)")
    p.add_argument("--db", default=None, help="override the SQLite path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("doctor", help="environment / paths / dependency report")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

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


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console()
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.config:
        load_config(args.config)
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
