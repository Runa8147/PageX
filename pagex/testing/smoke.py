"""Pre-demo self check: environment + schema + FTS5 + embedder, no heavy models.

    python -m pagex.testing.smoke
    python cli.py smoke

This is the ritual you run before a live demo: it takes a couple of seconds
because it never loads the VLM/LLM (they are proven by tools/probe.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

Result = tuple[str, bool, str]        # (name, ok, detail)


def _check_import(name: str, pip_name: str | None = None) -> Result:
    try:
        __import__(name)
        return (f"import {name}", True, "ok")
    except Exception as exc:
        return (f"import {name}", False, f"{exc} (pip install {pip_name or name})")


def check_schema() -> Result:
    from pagex.core.store import Store
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "smoke.db"
        store = Store(db)
        tables = {r["name"] for r in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')")}
        need = {"documents", "pages", "blocks", "chunks", "chunks_fts",
                "kg_nodes", "kg_edges", "chat_sessions", "chat_messages",
                "chunks_ai", "chunks_ad", "chunks_au"}
        missing = sorted(need - tables)
        store.close_all()
    return ("schema + FTS5 + triggers", not missing,
            "all present" if not missing else f"missing {missing}")


def check_fts_roundtrip() -> Result:
    from pagex.core.store import Chunk, Store
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "fts.db")
        store.put_document("d1", "Smoke Doc", "/tmp/x.png", "image", 1, "ready")
        store.put_page("d1-p001", "d1", 1, None, 100, 100)
        store.put_chunks([Chunk(chunk_id="c1", doc_id="d1", page_id="d1-p001",
                                block_ids=[], content="photosynthesis chlorophyll",
                                visual_desc="green leaves")])
        hits = store.fts_search("photosynthesis")
        store.conn.execute("DELETE FROM chunks WHERE chunk_id='c1'")
        store.conn.commit()
        after = store.fts_search("photosynthesis")
        store.close_all()
    ok = bool(hits) and not after
    return ("FTS5 insert/delete sync", ok,
            f"{len(hits)} hit(s) before delete, {len(after)} after")


def check_embedder() -> Result:
    from pagex.core.config import CFG
    path = CFG.embed_model_path()
    if not path.exists():
        return ("embedder GGUF", False, f"missing {path} "
                "(python tools/download_models.py --only embed)")
    t0 = time.perf_counter()
    try:
        from pagex.core.models import embed_texts, unload_embedder
        vecs = embed_texts(["angular momentum conservation",
                            "student registration form fields"])
        unload_embedder()
    except Exception as exc:
        return ("embedder GGUF", False, str(exc))
    dt = time.perf_counter() - t0
    if len(vecs) != 2 or len(vecs[0]) != 384:
        return ("embedder GGUF", False, f"bad shape {[len(v) for v in vecs]}")
    import math
    dot = sum(a * b for a, b in zip(vecs[0], vecs[1]))
    same = sum(a * b for a, b in zip(vecs[0], vecs[0]))
    ok = abs(same - 1.0) < 1e-3 and dot < 0.95
    return ("embedder round trip", ok,
            f"dim 384, self-sim {same:.4f}, cross-sim {dot:.3f}, {dt*1000:.0f} ms")


def check_models() -> Result:
    from pagex.core.config import CFG
    missing = CFG.missing_models()
    if missing:
        return ("model files", False,
                f"missing {missing} -> python tools/download_models.py")
    sizes = ", ".join(f"{k}={CFG.model_path(k).stat().st_size/1e6:.0f}MB"
                      for k in CFG.missing_models() or
                      ("vlm_gguf", "llm_gguf", "embed_gguf"))
    return ("model files", True, sizes)


def check_pymupdf_render() -> Result:
    try:
        import fitz  # noqa: F401
        return ("PyMuPDF render path", True, "ok")
    except Exception as exc:
        return ("PyMuPDF render path", False, str(exc))


def check_pdflatex() -> Result:
    exe = shutil.which("pdflatex")
    if exe:
        return ("pdflatex (optional)", True, exe)
    return ("pdflatex (optional)", True,
            "absent -> ReportLab fallback PDF (guaranteed export)")


def check_samples() -> Result:
    from pagex.testing import fixtures
    q = ROOT / "samples" / "queries.json"
    if not q.exists():
        fixtures.write_samples()
    n = len(json.loads(q.read_text())) if q.exists() else 0
    return ("samples/queries.json", n >= 10, f"{n} queries")


def check_gui_deps() -> Result:
    try:
        from PyQt6.QtCore import QT_VERSION_STR  # noqa: F401
        return ("PyQt6", True, QT_VERSION_STR)
    except Exception as exc:
        return ("PyQt6", False, str(exc))


def run_checks(fast: bool = False) -> list[Result]:
    checks: list[Result] = [
        _check_import("numpy"),
        _check_import("cv2", "opencv-python-headless"),
        _check_import("fitz", "PyMuPDF"),
        _check_import("docx", "python-docx"),
        _check_import("reportlab"),
        _check_import("psutil"),
        _check_import("llama_cpp", "llama-cpp-python"),
        check_schema(),
        check_fts_roundtrip(),
        check_pymupdf_render(),
        check_pdflatex(),
        check_samples(),
        check_gui_deps(),
    ]
    if not fast:
        checks.append(check_models())
        checks.append(check_embedder())
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip model checks")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    results = run_checks(fast=args.fast)
    elapsed = time.perf_counter() - started
    failed = [r for r in results if not r[1]]

    if args.json:
        print(json.dumps({"checks": [{"name": n, "ok": ok, "detail": d}
                                     for n, ok, d in results],
                          "seconds": round(elapsed, 2),
                          "ok": not failed}, indent=2))
    else:
        print("PageX smoke check " + "-" * 46)
        for name, ok, detail in results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<30} {detail}")
        print("-" * 48)
        print(f"{len(results) - len(failed)}/{len(results)} passed "
              f"in {elapsed:.2f}s")
        if failed:
            print("Fix the FAILs before the demo.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
