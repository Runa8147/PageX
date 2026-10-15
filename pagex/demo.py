"""``pagex demo`` - a stranger can evaluate the project in under two minutes.

One command does three things:

1. materialises the bundled/synthetic sample corpus (no download, no internet),
2. ingests one small sample page (or every page with ``--all``),
3. prints five suggested queries plus the exact follow-up commands.

Re-running is cheap: a document whose source is already indexed ``ready`` is
skipped instead of being re-recognised (a VLM call costs ~90 s/page on the
reference laptop).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from .core import paths

log = logging.getLogger(__name__)

FALLBACK_QUERIES = [
    "What is the moment of inertia of a solid cylinder?",
    "How does torque relate to angular acceleration?",
    "state the chain rule for derivatives",
    "angular momentum conservation",
    "measured energy table with mass and time",
]

DIGITAL_SAMPLE = "digital/page_sample.pdf"
PREFERRED_SAMPLES = ["notes/small_note.png", "notes/1_rotational_motion.png"]


def locate_samples() -> Path:
    """Folder with ``queries.json`` + sample pages (generated when absent)."""
    folder = paths.samples_dir(create=True)
    if not (folder / "queries.json").exists():
        from .testing import fixtures
        try:
            fixtures.write_samples(folder)
        except Exception as exc:                   # pragma: no cover
            log.warning("could not generate samples in %s: %s", folder, exc)
    return folder


def suggested_queries(folder: Path, limit: int = 5) -> list[str]:
    try:
        data = json.loads((folder / "queries.json").read_text(encoding="utf-8"))
        queries = [str(q.get("query", "")).strip() for q in data if q.get("query")]
        queries = [q for q in queries if q][:limit]
        if queries:
            return queries
    except Exception:
        pass
    return FALLBACK_QUERIES[:limit]


def demo_targets(folder: Path, all_samples: bool = False) -> list[tuple[Path, bool]]:
    """``(path, handwritten)`` pairs to ingest, cheapest first."""
    notes = sorted((folder / "notes").glob("*.png"))
    by_name = {p.name: p for p in notes}
    targets: list[tuple[Path, bool]] = []
    if all_samples:
        targets += [(p, True) for p in notes]
    else:
        for name in PREFERRED_SAMPLES:
            if name in by_name:
                targets.append((by_name[name], True))
                break
        if not targets and notes:
            targets.append((notes[0], True))
    pdf = folder / DIGITAL_SAMPLE
    if all_samples and pdf.exists():
        targets.append((pdf, False))
    return targets


def run_demo(*, ingest: bool = True, all_samples: bool = False,
             db: str | None = None, json_out: bool = False) -> int:
    from .core.store import Store

    folder = locate_samples()
    queries = suggested_queries(folder)
    targets = demo_targets(folder, all_samples)
    plan = [{"path": str(p), "handwritten": hw} for p, hw in targets]

    if not ingest:
        payload = {"samples_dir": str(folder), "targets": plan, "queries": queries}
        if json_out:
            print(json.dumps(payload, indent=2))
        else:
            print(f"samples: {folder}")
            for item in plan:
                print(f"  would ingest {item['path']} (handwritten={item['handwritten']})")
            print("queries:")
            for q in queries:
                print(f"  - {q}")
        return 0

    if not targets:
        print("no sample pages found - run `python -m pagex.testing.fixtures`")
        return 1

    from .core.pipeline import ingest as ingest_path

    store = Store(db) if db else Store()
    failures = 0
    seeded: list[str] = []
    for path, handwritten in targets:
        existing = store.find_document_by_source(str(path.resolve()))
        if existing and existing.get("status") == "ready":
            print(f"[skip] {path.name} already indexed ({existing['doc_id']})")
            seeded.append(existing["doc_id"])
            continue
        print(f"[ingest] {path.name} ({'handwritten' if handwritten else 'digital'}) ...")

        def progress(stage: str, cur: int, total: int, _n: str = path.name) -> None:
            print(f"    {stage:<11} {cur}/{total}")

        try:
            report = ingest_path(path, handwritten=handwritten, store=store,
                                 progress=progress)
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {exc}")
            continue
        if report.ok:
            seeded.append(report.doc_id)
            print(f"    -> {report.n_pages} page(s), {report.n_blocks} blocks, "
                  f"{report.n_chunks} chunks in {report.seconds:.1f}s")
        else:
            failures += 1
            print(f"    FAILED: {'; '.join(report.errors) or 'no blocks'}")
    store.close_all()

    if json_out:
        print(json.dumps({"seeded": seeded, "queries": queries,
                          "samples_dir": str(folder)}, indent=2))
        return 1 if failures else 0

    print("\nSample corpus ready. Try these queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. pagex search \"{q}\"")
    print("\nNext steps:")
    print('  pagex ask "What is the moment of inertia?"    # grounded answer + citations')
    print("  pagex chat                                   # multi-turn, offline")
    print("  pagex gui                                    # desktop shell")
    if failures:
        print(f"\n{failures} sample(s) failed - run `pagex doctor` and check "
              "`pagex models download`.")
        return 1
    return 0
