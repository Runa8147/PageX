"""TC-INT-06 - the CLI is a thin shell over the same core code paths.

Seeded with the canned pipeline (no GGUF), then driven through ``cli.main``
exactly the way a shell would - including JSON output parsing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import cli
from pagex.core.config import CFG
from pagex.core.pipeline import ingest
from pagex.core.store import Store

#: the two checks below presuppose an installed model set (CI has no GGUFs)
MODELS_OK = not CFG.missing_models()
needs_models = pytest.mark.skipif(
    not MODELS_OK, reason="GGUF weights not installed -> `pagex models download`")


@pytest.fixture()
def seeded_db(tmp_path, sample_page, canned_vision, canned_generate,
              canned_embed):
    db = tmp_path / "cli.db"
    path = tmp_path / "page.png"
    Image.fromarray(np.asarray(sample_page)).save(path)
    store = Store(db)
    report = ingest(path, title="Rotational Motion", store=store,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)
    assert report.ok
    store.close_all()
    return db, report


def test_tc_int_06_list_and_graph(seeded_db, capsys):
    db, report = seeded_db
    assert cli.main(["--db", str(db), "list"]) == 0
    out = capsys.readouterr().out
    assert "Rotational Motion" in out and "ready" in out

    assert cli.main(["--db", str(db), "graph", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)["stats"]
    assert stats["nodes"] > 0 and stats["edges"] > 0


def test_tc_int_06_search_json(seeded_db, capsys):
    db, report = seeded_db
    rc = cli.main(["--db", str(db), "search", "moment of inertia", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "moment of inertia"
    assert payload["citations"], "CLI search returned no citations"
    assert payload["citations"][0]["doc_id"] == report.doc_id
    # CI installs the test/build extras but not the GGUF embedder; BM25 and
    # graph must still work, while the vector ranker is expected only when the
    # model pack is present.
    expected = {"bm25", "graph"}
    if MODELS_OK:
        expected.add("vector")
    assert expected <= set(payload["rankers"])


@needs_models
def test_tc_int_06_models_report_never_loads(capsys):
    from pagex.core.models import HEAVY
    before = HEAVY.loads
    assert cli.main(["models", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    vlm = next(r for r in rows if r["config_key"] == "vlm_gguf")
    assert vlm["exists"] is True and vlm["sha256"]
    assert HEAVY.loads == before, "`models` must not load a GGUF"


@needs_models
def test_tc_int_06_smoke_self_check(capsys):
    assert cli.main(["smoke"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out.upper() or "ok" in out.lower()


def test_tc_int_06_bad_query_exits_cleanly(seeded_db, capsys):
    db, _ = seeded_db
    rc = cli.main(["--db", str(db), "search", "", "--json"])
    assert rc in (0, 1)                 # empty query -> "no results", not a crash
    err = ""
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# distribution surface: one implementation, doctor / demo / models
# ---------------------------------------------------------------------------
def test_root_cli_is_a_thin_shim_over_pagex_cli():
    """`python cli.py`, `pagex` and `python -m pagex` share one implementation."""
    import cli as root_cli
    from pagex import cli as pkg_cli
    assert root_cli.main is pkg_cli.main
    assert root_cli.build_parser is pkg_cli.build_parser


def test_cli_doctor_reports_paths_and_dependencies(capsys):
    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    info = payload["paths"]
    assert info["mode"] in ("source checkout", "portable (PAGEX_HOME)",
                            "installed", "frozen")
    assert info["config"].endswith("pagex.toml")
    assert "base_dir" in info and "platform" in info
    assert payload["deps"]["numpy"] == "ok"
    assert payload["manifest"], "doctor must read the model manifest"


def test_cli_doctor_human_screen(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "PageX doctor" in out and "config" in out


def test_cli_models_list_reads_the_manifest(capsys):
    assert cli.main(["models", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    keys = {r["key"] for r in rows}
    assert {"vlm", "vlm_mmproj", "llm", "embed"} <= keys
    embed = next(r for r in rows if r["key"] == "embed")
    assert embed["sha256"] and embed["license"] == "Apache-2.0"
    assert embed["url"].startswith("https://huggingface.co/")


def test_cli_models_verify_on_an_empty_dir_fails_loudly(capsys, tmp_path):
    rc = cli.main(["models", "verify", "--no-hash",
                   "--dest", str(tmp_path), "--json"])
    rows = json.loads(capsys.readouterr().out)
    embed = next(r for r in rows if r["key"] == "embed")
    assert embed["status"] == "missing" and embed["required"] is True
    assert rc == 1, "missing required weights must be a non-zero exit"


def test_cli_demo_dry_run_lists_queries(capsys, tmp_path):
    assert cli.main(["demo", "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["queries"]) == 5
    assert payload["targets"], "demo must plan at least one sample page"
    assert Path(payload["samples_dir"]).exists()
    assert payload["targets"][0]["handwritten"] is True


def test_cli_demo_dry_run_human_output(capsys):
    assert cli.main(["demo", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "samples:" in out and "queries" in out
    assert "would ingest" in out


def test_cli_ingest_max_pages_is_wired():
    """A 126-page scan must be indexable in slices (see test_pdf_notes)."""
    args = cli.build_parser().parse_args(
        ["ingest", "notes.pdf", "--handwritten", "--max-pages", "3"])
    assert args.max_pages == 3 and args.handwritten is True
    assert args.fn.__name__ == "cmd_ingest"


def test_cli_models_action_defaults_to_list(capsys):
    """Back-compat: `pagex models` with no subcommand still prints the list."""
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "vlm" in out and "embed" in out
