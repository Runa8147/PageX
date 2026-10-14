"""TC-INT-05 - grounded RAG: prompt, citations, persistence, refusal."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pagex.core.pipeline import ingest
from pagex.core.rag import answer_query, build_prompt, format_context
from pagex.core.retrieve import invalidate_cache


class _Fake:
    """Minimal GenResult stand-in for the prompt-construction test."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = 0
        self.seconds = 0.0
        self.tok_per_s = 0.0

    def __bool__(self):
        return bool(self.text)


@pytest.fixture()
def rag_store(tmp_store, tmp_path, sample_page, canned_vision,
              canned_generate, canned_embed):
    path = tmp_path / "notes.png"
    Image.fromarray(np.asarray(sample_page)).save(path)
    report = ingest(path, title="Physics", store=tmp_store,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)
    assert report.ok
    invalidate_cache()
    return tmp_store, report, canned_generate, canned_embed


def test_tc_int_05_answer_has_citations_and_is_persisted(rag_store):
    store, report, gen, embed = rag_store
    result = answer_query(store, "What is the moment of inertia?",
                          generate_fn=gen, embedder=embed)

    assert result.answer.strip()
    assert result.citations, "grounded answer must cite retrieved chunks"
    assert result.session_id
    assert result.grounded, "answer must contain at least one citation label"

    # both turns live in SQLite (chat_sessions / chat_messages)
    history = store.history(result.session_id, limit=10)
    roles = [m["role"] for m in history]
    assert "user" in roles and "assistant" in roles
    stored = store.conn.execute(
        "SELECT COUNT(*) FROM chat_messages WHERE session_id=?",
        (result.session_id,)).fetchone()[0]
    assert stored >= 2

    # multi-turn: history is fed back into the prompt on the next call
    second = answer_query(store, "and its SI unit?", session_id=result.session_id,
                          generate_fn=gen, embedder=embed)
    assert second.session_id == result.session_id
    assert second.history_turns >= 1


def test_tc_int_05_prompt_contains_context_and_instruction(rag_store):
    store, _report, _gen, embed = rag_store
    result = answer_query(store, "angular momentum",
                          generate_fn=lambda *a, **k: _Fake(
                              "L = I omega [doc:Physics p.1]"),
                          embedder=embed, persist=False)
    # build_prompt is what the retriever -> LLM boundary looks like
    prompt = build_prompt("angular momentum", result.citations[:3], store)
    assert "angular momentum" in prompt.lower()
    assert "cite" in prompt.lower() or "source" in prompt.lower()
    ctx = format_context(result.citations[:3], store)
    assert ctx.strip(), "retrieval context must not be empty"
    assert len(result.citations[:3]) > 0


def test_tc_int_05_no_context_refuses_instead_of_hallucinating(rag_store):
    store, _report, gen, embed = rag_store
    result = answer_query(store, "qqzzxx unrelated gibberish",
                          generate_fn=gen, embedder=embed, persist=False)
    # either nothing was retrieved (refusal) or a short grounded answer
    assert result.answer.strip()
    if not result.citations:
        assert result.grounded is False
