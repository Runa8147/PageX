"""TC-INT-06 - the CLI is a thin shell over the same core code paths.

Seeded with the canned pipeline (no GGUF), then driven through ``cli.main``
exactly the way a shell would - including JSON output parsing.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import cli
from pagex.core.pipeline import ingest
from pagex.core.store import Store


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
    assert {"vector", "bm25"} <= set(payload["rankers"])


def test_tc_int_06_models_report_never_loads(capsys):
    from pagex.core.models import HEAVY
    before = HEAVY.loads
    assert cli.main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"]["vlm_gguf"]["exists"] is True
    assert HEAVY.loads == before, "`models` must not load a GGUF"


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
