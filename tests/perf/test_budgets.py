"""TC-PERF-01..03 - the numbers the manual gates the demo on.

* TC-PERF-01  p95 hybrid-search latency < 3 s (embed + FTS + vector + RRF)
* TC-PERF-02  worst-case RSS stays under the 6.2 GB contract, release works
* TC-PERF-03  throughput: embedder >= 30 texts/s, canned ingest >= 5 blocks/s
"""
from __future__ import annotations

import statistics
import time

import pytest

from pagex.core.config import CFG
from pagex.core.store import Chunk, Store

QUERIES = [
    "moment of inertia of a solid cylinder",
    "angular momentum conservation law",
    "why does torque cause angular acceleration",
    "kinetic energy of a rotating body",
    "relationship between torque and angular acceleration",
    "figure skater pulls arms and spins faster",
    "definition of rotational dynamics",
    "sum of m_i times r_i squared",
    "work and energy in rotation",
    "unit of angular momentum",
]

MODELS_OK = CFG.model_path("embed_gguf").exists()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _seed_chunks(store: Store, n: int = 300, embed_fn=None) -> None:
    """Synthetic corpus - perf must not depend on the VLM being run."""
    store.put_document("perf-doc", "Perf Corpus", "/tmp/perf.png", "image",
                       n, "ready")
    store.put_page("perf-doc-p001", "perf-doc", 1, None, 1000, 1400)
    texts = [
        "the moment of inertia depends on how mass is distributed "
        "around the axis of rotation",
        "angular momentum is conserved when no external torque acts",
        "torque equals moment of inertia times angular acceleration",
        "rotational kinetic energy is one half I omega squared",
        "a figure skater pulls arms in and spins faster",
        "power in rotation is torque times angular velocity",
        "the parallel axis theorem shifts the inertia about a new point",
        "linear momentum of each mass element sums to p equals m v",
    ]
    vecs = embed_fn([t for t in texts for _ in range(n // len(texts) + 1)]) \
        if embed_fn else None
    chunks = []
    for i in range(n):
        text = texts[i % len(texts)] + f" variant {i}"
        vec = vecs[i] if vecs else None
        chunks.append(Chunk(
            chunk_id=f"perf-doc-p001-c{i:04d}", doc_id="perf-doc",
            page_id="perf-doc-p001", block_ids=[f"b{i}"], content=text,
            visual_desc="perf fixture", vec=vec, vec_dim=384 if vec else 0,
            token_count=len(text) // 4))
    store.put_chunks(chunks)


def _p95(samples: list[float]) -> float:
    if len(samples) < 2:
        return samples[0] if samples else 0.0
    return float(statistics.quantiles(samples, n=20)[18])   # 95th percentile


# ---------------------------------------------------------------------------
# TC-PERF-01
# ---------------------------------------------------------------------------
def test_tc_perf_01_search_p95_under_3s(tmp_store, canned_embed):
    from pagex.core.retrieve import invalidate_cache, search

    embed_fn = None
    if MODELS_OK:
        from pagex.core.models import embed_texts
        embed_fn = embed_texts
        embed_fn(["warm up the embedder"])           # one-time load cost
    embedder = embed_fn or canned_embed
    # seed the corpus in the SAME vector space queries will use
    _seed_chunks(tmp_store, 300, embed_fn=embedder)
    invalidate_cache()

    # warm the FTS/VM page cache and the cosine path
    for q in QUERIES[:3]:
        search(tmp_store, q, embedder=embedder)

    samples: list[float] = []
    for _ in range(3):
        for q in QUERIES:
            t0 = time.perf_counter()
            result = search(tmp_store, q, embedder=embedder)
            samples.append((time.perf_counter() - t0) * 1000)
            assert result.citations, f"no results for {q!r}"

    p95 = _p95(samples)
    median = statistics.median(samples)
    print(f"\nsearch latency: median {median:.0f} ms, p95 {p95:.0f} ms "
          f"over {len(samples)} queries "
          f"({'real embedder' if embed_fn else 'canned embedder'})")
    assert p95 < 3000, f"p95 search {p95:.0f} ms >= 3000 ms gate"


# ---------------------------------------------------------------------------
# TC-PERF-02
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_tc_perf_02_ram_contract(tmp_store):
    pytest.importorskip("psutil")
    from pagex.core.models import HEAVY, rss_gb

    if not CFG.model_path("vlm_gguf").exists() or not CFG.model_path("llm_gguf").exists():
        pytest.skip("GGUF models not downloaded")

    budget = 6.2
    peak = 0.0
    HEAVY.release()
    baseline = rss_gb()

    HEAVY.acquire("vlm", CFG.model_path("vlm_gguf"),
                  mmproj=CFG.model_path("vlm_mmproj"), n_ctx=CFG.vlm_ctx)
    peak = max(peak, rss_gb())
    assert peak < budget, f"VLM resident {peak:.2f} GB >= {budget} GB"

    HEAVY.acquire("llm", CFG.model_path("llm_gguf"), n_ctx=CFG.llm_ctx)
    peak = max(peak, rss_gb())
    assert HEAVY.key == "llm"
    assert peak < budget, f"worst-case RSS {peak:.2f} GB >= {budget} GB"

    HEAVY.release()
    after = rss_gb()
    assert after < peak - 1.0, "release must return the RAM"
    print(f"\nRAM: baseline {baseline:.2f} -> peak {peak:.2f} -> "
          f"after release {after:.2f} GB (budget {budget} GB)")


# ---------------------------------------------------------------------------
# TC-PERF-03
# ---------------------------------------------------------------------------
def test_tc_perf_03_embed_throughput(canned_embed):
    texts = [f"throughput sample sentence number {i}" for i in range(200)]
    embed_fn = canned_embed
    if MODELS_OK:
        from pagex.core.models import embed_texts
        embed_fn = embed_texts
    t0 = time.perf_counter()
    vecs = embed_fn(texts)
    dt = time.perf_counter() - t0
    rate = len(texts) / dt
    assert len(vecs) == 200 and len(vecs[0]) == 384
    floor = 30.0 if embed_fn is not canned_embed else 100.0
    assert rate >= floor, f"embedder only {rate:.1f} texts/s (< {floor})"
    print(f"\nembed: {rate:.1f} texts/s, dim=384")


def test_tc_perf_03_canned_ingest_throughput(tmp_store, tmp_path, sample_page,
                                             canned_vision, canned_generate,
                                             canned_embed):
    import numpy as np
    from PIL import Image

    from pagex.core.pipeline import ingest

    paths = []
    for i in range(3):
        p = tmp_path / f"p{i}.png"
        Image.fromarray(np.asarray(sample_page)).save(p)
        paths.append(p)

    t0 = time.perf_counter()
    blocks = 0
    for p in paths:
        report = ingest(p, store=tmp_store, vision_fn=canned_vision,
                        generate_fn=canned_generate, embed_fn=canned_embed)
        assert report.ok
        blocks += report.n_blocks
    dt = time.perf_counter() - t0
    rate = blocks / dt
    assert rate >= 5.0, f"pipeline only {rate:.1f} blocks/s (excluding model time)"
    print(f"\ningest pipeline: {rate:.1f} blocks/s with canned models")
