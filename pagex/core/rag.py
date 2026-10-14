"""WP8 - grounded RAG chat: answer *only* from retrieved context.

Hallucination resistance is an explicit quality gate: the system prompt
forbids answering outside CONTEXT and forces ``[doc:title p.N]`` citations,
and :func:`answer_query` records whether at least one citation survived into
the final answer.

History is trimmed (last 6 turns / ~1500 tokens) so a long session cannot
push the KV cache past ``llm_ctx`` on an 8 GB machine.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import CFG
from .models import GenResult, embed_texts, generate
from .retrieve import SearchResult, search
from .store import Citation, Store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PageX, a document assistant running fully offline.
Answer ONLY from the CONTEXT blocks provided. Rules:
1. If the context does not contain the answer, say "Not found in the indexed documents." - never guess.
2. Cite every factual claim with its source as [doc:TITLE p.N].
3. Keep the same language as the question. Be concise and technical.
4. For equations, reproduce the LaTeX from the context exactly.
5. You may use the conversation history only to resolve pronouns, never as a source of new facts."""

MAX_TURNS = 6
MAX_HISTORY_TOKENS = 1500


@dataclass
class RagResult:
    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    session_id: str = ""
    grounded: bool = False                 # >=1 citation actually appeared
    gen: GenResult | None = None
    search: SearchResult | None = None
    seconds: float = 0.0
    history_turns: int = 0

    @property
    def cited_labels(self) -> list[str]:
        return [f"[doc:{c.title} p.{c.page_no}]" for c in self.citations]

    def to_dict(self) -> dict:
        return {"query": self.query, "answer": self.answer,
                "citations": [c.to_dict() for c in self.citations],
                "session_id": self.session_id, "grounded": self.grounded,
                "seconds": round(self.seconds, 3),
                "tok_per_s": self.gen.tok_per_s if self.gen else 0.0}


# ---------------------------------------------------------------------------
# context assembly
# ---------------------------------------------------------------------------
def format_context(citations: Sequence[Citation], store: Store | None = None,
                   max_chars: int = 4000) -> str:
    """Numbered CONTEXT blocks the prompt refers to."""
    parts: list[str] = []
    used = 0
    for i, c in enumerate(citations, start=1):
        body = c.snippet
        if store is not None:
            row = store.conn.execute(
                "SELECT content FROM chunks WHERE chunk_id=?", (c.chunk_id,)).fetchone()
            if row:
                body = row["content"]
        block = (f"[{i}] doc={c.title!r} page={c.page_no} "
                 f"blocks={','.join(c.block_ids)}\n{body}")
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_prompt(query: str, citations: Sequence[Citation],
                 store: Store | None = None) -> str:
    context = format_context(citations, store)
    if not context:
        context = "(no matching documents were retrieved)"
    return (f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (cite as [doc:title p.N]):")


# ---------------------------------------------------------------------------
# history handling
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def trim_history(messages: Sequence[dict], max_turns: int = MAX_TURNS,
                 max_tokens: int = MAX_HISTORY_TOKENS) -> list[dict]:
    """Keep the most recent turns inside both budget (turns and tokens)."""
    kept: list[dict] = []
    tokens = 0
    for msg in reversed(list(messages)):
        if len(kept) >= max_turns * 2:
            break
        cost = estimate_tokens(str(msg.get("content", "")))
        if tokens + cost > max_tokens and kept:
            break
        kept.append(dict(msg))
        tokens += cost
    return list(reversed(kept))


def history_to_messages(history: Sequence[dict],
                        system: str = SYSTEM_PROMPT) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system}]
    for m in trim_history(history):
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        msgs.append({"role": role, "content": str(m.get("content", ""))})
    return msgs


# ---------------------------------------------------------------------------
# the main entry point
# ---------------------------------------------------------------------------
CITATION_RE = re.compile(r"\[doc:(.+?)\s+p\.?(\d+)\]")


def answer_query(store: Store, query: str, *,
                 session_id: str | None = None,
                 doc_ids: Sequence[str] | None = None,
                 history: Sequence[dict] | None = None,
                 k: int | None = None,
                 persist: bool = True,
                 generate_fn: Callable[..., GenResult] = generate,
                 embedder=embed_texts) -> RagResult:
    """retrieve -> prompt -> local LLM -> grounded answer + citations."""
    t0 = time.perf_counter()
    session_id = session_id or store.new_session()

    if history is None:
        history = [m for m in store.history(session_id, limit=MAX_TURNS * 2)]

    result = search(store, query, k=k, doc_ids=doc_ids, embedder=embedder)
    citations = result.top(k or CFG.top_k)

    messages = history_to_messages(history)
    messages.append({"role": "user", "content": build_prompt(query, citations, store)})

    answer = ""
    gen: GenResult | None = None
    if citations:
        try:
            gen = _chat(messages, generate_fn)
            answer = gen.text.strip()
        except Exception as exc:
            log.error("generation failed: %s", exc)
            answer = f"(model error: {exc})"
    else:
        answer = "Not found in the indexed documents."

    grounded = _is_grounded(answer, citations)
    if citations and not answer:
        answer = "Not found in the indexed documents."

    if persist:
        store.add_message(session_id, "user", query)
        store.add_message(session_id, "assistant", answer, citations)

    result_obj = RagResult(query=query, answer=answer, citations=citations,
                           session_id=session_id, grounded=grounded, gen=gen,
                           search=result, history_turns=len(history))
    result_obj.seconds = time.perf_counter() - t0
    return result_obj


def _chat(messages: list[dict], generate_fn) -> GenResult:
    """Run a full message list through the local LLM.

    ``generate_fn`` is injectable so tests never load a GGUF: it receives the
    OpenAI-style message list (first call) or the rendered prompt.
    """
    try:
        return generate_fn(messages)                 # type: ignore[call-arg]
    except TypeError:
        # fallback signature: generate(prompt, system=...)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")
        return generate_fn(user, system=system)


def _is_grounded(answer: str, citations: Sequence[Citation]) -> bool:
    """True when the answer cites at least one retrieved page."""
    labels = CITATION_RE.findall(answer)
    if not labels:
        return False
    wanted = {(c.title.strip(), c.page_no) for c in citations}
    for title, page in labels:
        if (title.strip(), int(page)) in wanted:
            return True
    # tolerate a title mismatch (shortened by the model) as long as page matches
    pages = {c.page_no for c in citations}
    return any(int(page) in pages for _t, page in labels)


# ---------------------------------------------------------------------------
# streaming-ish helper for the CLI / GUI (simple loop)
# ---------------------------------------------------------------------------
def chat(store: Store, session_id: str | None = None, *, k: int | None = None,
         generate_fn=generate, embedder=embed_texts) -> tuple[str, list[RagResult]]:
    """Interactive REPL used by ``cli.py chat``; returns (session_id, results)."""
    session_id = session_id or store.new_session()
    results: list[RagResult] = []
    print(f"PageX chat · session {session_id} · 'exit' to quit")
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break
        res = answer_query(store, query, session_id=session_id, k=k,
                           generate_fn=generate_fn, embedder=embedder)
        results.append(res)
        print(f"pagex> {res.answer}")
        if res.citations:
            print("  sources: " + ", ".join(res.cited_labels))
    return session_id, results


def summarize_citations(citations: Sequence[Citation], limit: int = 4) -> str:
    """Compact ``title p.N`` list for the GUI chat bubbles."""
    labels = []
    for c in citations[:limit]:
        labels.append(f"{c.title} p.{c.page_no}")
    extra = len(citations) - limit
    return ", ".join(labels) + (f" (+{extra})" if extra > 0 else "")
