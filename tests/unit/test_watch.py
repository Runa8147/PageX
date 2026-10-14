"""Watch-folder logic (self-ingest protection + debounce)."""
from __future__ import annotations

import time
from pathlib import Path

from pagex.core.watch import WatchState, is_candidate, pending, scan


def test_candidate_filters(tmp_path):
    (tmp_path / "note.png").write_bytes(b"x")
    (tmp_path / "~$note.png").write_bytes(b"x")
    (tmp_path / "partial.part").write_bytes(b"x")
    (tmp_path / "notes.tmp").write_bytes(b"x")
    (tmp_path / "readme.txt").write_text("x")
    (tmp_path / ".hidden.png").write_bytes(b"x")
    (tmp_path / "scan.pdf").write_bytes(b"%PDF")

    ok = {p.name for p in scan(tmp_path)}
    assert ok == {"note.png", "scan.pdf"}, ok


def test_artifacts_dir_is_never_ingested(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    inside = artifacts / "page_0001.png"
    inside.write_bytes(b"x")
    outside = tmp_path / "page_0001.png"
    outside.write_bytes(b"x")
    assert not is_candidate(inside, artifacts=artifacts)
    assert is_candidate(outside, artifacts=artifacts)


def test_debounce_waits_for_the_copy_to_finish(tmp_path):
    state = WatchState(tmp_path, debounce_s=0.2)
    f = tmp_path / "incoming.png"
    f.write_bytes(b"x" * 10)
    state.on_change(str(f))

    assert state.drain(now=time.time()) == [], "not debounced yet"
    time.sleep(0.25)
    ready = state.drain()
    assert [p.name for p in ready] == ["incoming.png"]

    # the same file is not ingested twice (self-ingest loop protection)
    state.on_change(str(f))
    time.sleep(0.25)
    assert state.drain() == []


def test_drain_now_ignores_missing_and_seen(tmp_path):
    state = WatchState(tmp_path, debounce_s=0.0)
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    state.on_change(str(f))
    assert [p.name for p in state.drain_now()] == ["a.png"]
    state.on_change(str(f))
    assert state.drain_now() == []
    state.on_change(str(tmp_path / "gone.png"))
    assert state.drain_now() == []


def test_on_change_ignores_non_candidates(tmp_path):
    state = WatchState(tmp_path, debounce_s=0.0)
    state.on_change(str(tmp_path / "notes.tmp"))
    assert state.drain_now() == []
    snap = state.snapshot()
    assert snap["pending"] == 0


def test_pending_helper(tmp_path):
    a = tmp_path / "a.png"
    a.write_bytes(b"x")
    seen = set()
    found = pending([a], seen=seen)
    assert found == [a]
    seen.add(str(a.resolve()))
    assert pending([a], seen=seen) == []
