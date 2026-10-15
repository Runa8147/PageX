#!/usr/bin/env python3
"""Hardware probe -> ``bench.json`` -> automatic performance tier.

Run this *before* writing any feature (build manual section 4.5): it measures
the machine you are actually on and records the numbers the report quotes.

    pagex bench              # embed + threads + LLM + VLM (default)
    pagex bench --fast       # skip the VLM phase
    pagex bench --full       # measure a real full-page VLM call
    python -m pagex.probe    # identical entry point

Outputs ``bench.json`` next to the active config:
    n_threads, tier (1|2|3), embed_texts/s, llm tok/s, VLM s/page, peak RSS.

Tier ladder (documented degradation):
  Tier-1  LLM >= 6 tok/s and VLM <= 60 s/page  -> defaults from pagex.toml
  Tier-2  otherwise                            -> page long-edge 768 px,
                                                  2048 VLM tokens, 1.5B chat LLM
  Tier-3  still slower                        -> page splitting (half-page calls)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import models
from .core.config import CFG, setup_logging
from .core import paths

PROMPT_30 = ("Explain in one sentence why the sky is blue. "
             "Keep it under thirty words total.")
VLM_PROBE_TOKENS = 200
VLM_PAGE_TOKENS = 600       # measured: caption + 9 regions ~= 521 tokens @ 768px


def avx2_present() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
        return "avx2" in flags.split("\n", 1)[-1] or "avx2" in flags
    except Exception:
        return "avx2" in platform.processor().lower()


def thread_candidates() -> list[int]:
    cpu = __import__("os").cpu_count() or 2
    return sorted({1, 2, min(4, cpu), min(8, cpu)})


def bench_embed() -> dict:
    texts = [
        "The moment of inertia depends on mass distribution.",
        "Angular momentum is conserved when no torque acts.",
        "Latent semantic structure survives LaTeX round-tripping.",
        "Hybrid search combines BM25 keywords with cosine similarity.",
        "Regenerative braking converts kinetic energy into stored charge.",
        "The derivative measures instantaneous rate of change.",
    ] * 4
    models.unload_embedder()
    t0 = time.perf_counter()
    vecs = models.embed_texts(texts)
    dt = time.perf_counter() - t0
    models.unload_embedder()
    return {"texts": len(texts), "seconds": round(dt, 3),
            "texts_per_s": round(len(texts) / dt, 1),
            "dim": len(vecs[0]) if vecs else 0,
            "rss_gb": round(models.rss_gb(), 3)}


def bench_llm(threads: int, tokens: int = 80, repeats: int = 2) -> dict:
    """Warm-up + best-of-*repeats* completion speed at *threads*.

    A single 40-token sample swings by +-30% on a laptop that is also running
    a desktop, so we warm the KV cache first and keep the best run.
    """
    CFG.n_threads = threads
    from .core.models import HEAVY
    HEAVY.release()
    t_load0 = time.perf_counter()
    # Measure the *primary* model: tier must reflect hardware capability, not
    # the tier-swapped fallback (otherwise slow -> fallback -> slow is circular).
    primary = CFG.model_path("llm_gguf")
    target = primary if primary.exists() else CFG.active_llm
    m = HEAVY.acquire("llm", target, n_ctx=CFG.llm_ctx)
    load_s = time.perf_counter() - t_load0

    # warm-up (page cache, repacked tensors, sampler)
    m.create_chat_completion([{"role": "user", "content": PROMPT_30}],
                             max_tokens=16, temperature=0.0)

    best = {"tok_per_s": 0.0, "gen_s": 0.0, "tokens": 0}
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = m.create_chat_completion(
            [{"role": "user", "content": PROMPT_30}],
            max_tokens=tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        n = int(r["usage"]["completion_tokens"])
        tok_s = (n / dt) if dt else 0.0
        if tok_s > best["tok_per_s"]:
            best = {"tok_per_s": tok_s, "gen_s": dt, "tokens": n}

    rss = models.rss_gb()
    HEAVY.release()
    return {"threads": threads, "load_s": round(load_s, 2),
            "gen_s": round(best["gen_s"], 2), "tokens": best["tokens"],
            "tok_per_s": round(best["tok_per_s"], 2),
            "rss_gb_after_load": round(rss, 2),
            "model": target.name}


def pick_threads(results: list[dict]) -> int:
    """Best tok/s, but never more than 5% worse than the peak with fewer cores."""
    if not results:
        return CFG.effective_threads
    best = max(r["tok_per_s"] for r in results)
    ok = [r for r in results if r["tok_per_s"] >= 0.95 * best]
    ok.sort(key=lambda r: r["threads"])
    return ok[0]["threads"]


def sample_page_path() -> Path:
    from .testing import fixtures
    folder = paths.samples_dir(create=True)
    sample = folder / "notes" / "1_rotational_motion.png"
    if not sample.exists():
        fixtures.write_samples()
        sample = folder / "notes" / "1_rotational_motion.png"
    return sample


def bench_vlm(sample: Path, fast: bool = False, full: bool = False) -> dict:
    if fast:
        return {"skipped": True}
    if not sample.exists():
        from pagex.testing import fixtures
        fixtures.write_samples()
    if not sample.exists():
        return {"error": f"missing sample {sample}"}
    from .core.preprocess import load_image, preprocess_page
    from .core.models import HEAVY

    img = load_image(sample)
    pre = preprocess_page(img)
    HEAVY.release()
    t_load0 = time.perf_counter()
    try:
        HEAVY.acquire("vlm", CFG.model_path("vlm_gguf"),
                      mmproj=CFG.model_path("vlm_mmproj"), n_ctx=CFG.vlm_ctx)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    load_s = time.perf_counter() - t_load0
    from .core.parse import PAGE_PROMPT
    # Run A: 8-token completion isolates the fixed per-page overhead
    # (clip encode + LLM prefill). Run B adds only decode time, so the
    # per-page estimate = overhead + tokens * decode_rate (unbiased).
    t0 = time.perf_counter()
    res_a = models.vision_generate(pre["image"], PAGE_PROMPT, max_tokens=8)
    dt_a = time.perf_counter() - t0
    tokens = VLM_PAGE_TOKENS if full else VLM_PROBE_TOKENS
    t0 = time.perf_counter()
    res = models.vision_generate(pre["image"], PAGE_PROMPT, max_tokens=tokens)
    dt = time.perf_counter() - t0
    rss = models.rss_gb()
    tok_s = res.tokens / dt if dt else 0.0
    n_delta = max(1, res.tokens - res_a.tokens)
    decode_s_per_tok = max(0.0, dt - dt_a) / n_delta
    if full:
        per_page = dt
    else:
        per_page = dt_a + decode_s_per_tok * max(0, VLM_PAGE_TOKENS - res_a.tokens)
    HEAVY.release()
    try:
        shown = str(sample.relative_to(paths.samples_dir()))
    except ValueError:
        shown = sample.name
    return {"sample": shown,
            "load_s": round(load_s, 2), "tokens": res.tokens,
            "seconds": round(dt, 2),
            "tok_per_s": round(tok_s, 2),
            "overhead_s": round(dt_a, 2),
            "decode_s_per_tok": round(decode_s_per_tok, 4),
            "seconds_per_page_est": round(per_page, 1),
            "seconds_per_page_measured": round(dt, 2) if full else None,
            "rss_gb_after_load": round(rss, 2),
            "max_side": CFG.runtime_vlm_max_side}


def choose_tier(llm_tok_s: float, vlm_s_page: float) -> str:
    if llm_tok_s >= 6 and (vlm_s_page <= 60 or vlm_s_page == 0):
        return "1"
    if vlm_s_page <= 180 and llm_tok_s >= 3:
        return "2"
    return "3"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip the VLM phase")
    ap.add_argument("--full", action="store_true",
                    help="measure a real full-page VLM call (slow)")
    ap.add_argument("--out", default=None, help="where to write bench.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(level=30 if args.quiet else 20)

    sample = sample_page_path()
    out = Path(args.out) if args.out else paths.bench_path()
    bench: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "ram_total_gb": round(__import__("psutil").virtual_memory().total / 1e9, 1),
        "avx2": avx2_present(),
    }

    print("[1/4] embedder ...", end=" ", flush=True)
    try:
        bench["embed"] = bench_embed()
        print(f"{bench['embed']['texts_per_s']} texts/s, dim={bench['embed']['dim']}")
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        return 1

    print("[2/4] thread sweep + LLM tok/s ...")
    llm_runs = []
    for n in thread_candidates():
        try:
            run = bench_llm(n)
        except FileNotFoundError as exc:
            print(f"   threads={n}: {exc}")
            break
        llm_runs.append(run)
        print(f"   threads={n}: {run['tok_per_s']} tok/s "
              f"(load {run['load_s']}s, rss {run['rss_gb_after_load']} GB)")
    bench["llm_runs"] = llm_runs
    bench["n_threads"] = pick_threads(llm_runs)
    best = max(llm_runs, key=lambda r: r["tok_per_s"]) if llm_runs else None
    bench["llm"] = best or {"error": "no LLM run"}
    CFG.n_threads = bench["n_threads"]

    print(f"[3/4] VLM page probe ({'full' if args.full else 'fast'}) ...")
    vlm = bench_vlm(sample, fast=args.fast, full=args.full)
    bench["vlm"] = vlm
    if "error" in vlm:
        print(f"   {vlm['error']}")
    elif vlm.get("skipped"):
        print("   skipped (--fast)")
    else:
        print(f"   {vlm['tok_per_s']} tok/s -> ~{vlm['seconds_per_page_est']} s/page "
              f"(rss {vlm['rss_gb_after_load']} GB)")

    llm_tok_s = float(bench["llm"].get("tok_per_s") or 0.0)
    vlm_page = float(vlm.get("seconds_per_page_est") or 0.0)
    bench["tier"] = choose_tier(llm_tok_s, vlm_page)
    bench["rss_peak_gb"] = max(
        [float(bench["embed"].get("rss_gb", 0)),
         float(bench["llm"].get("rss_gb_after_load", 0) or 0),
         float(vlm.get("rss_gb_after_load", 0) or 0)])

    print(f"[4/4] writing {out} ...")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bench, indent=2), encoding="utf-8", newline="\n")
    print(f"\n{out}")
    print(json.dumps({k: bench[k] for k in ("n_threads", "tier", "avx2",
                                            "ram_total_gb")}, indent=2))
    print(f"LLM {llm_tok_s} tok/s · VLM ~{vlm_page} s/page · "
          f"embed {bench['embed']['texts_per_s']} texts/s · "
          f"peak RSS {bench['rss_peak_gb']} GB")
    if bench["tier"] != "1":
        print("NOTE: tier 2/3 -> the VLM page shrinks to 768px (ADR ladder); "
              "the chat model swaps to the 1.5B fallback only if "
              f"bench.llm.tok_per_s < 6 (now {llm_tok_s}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
