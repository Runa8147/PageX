#!/usr/bin/env python3
"""Fetch every GGUF the runtime needs into ./models (resumable).

ADR-002: all models are GGUF / Q4_K_M (or lighter) so they fit the 8 GB
contract together with the OS.  Nothing here is loaded at import time -
this script only *downloads*.

    python tools/download_models.py            # everything
    python tools/download_models.py --list     # show the manifest
    python tools/download_models.py --only vlm llm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (key, huggingface repo, filename, role, expected on-disk size in MB)
MANIFEST = [
    ("vlm",
     "lmstudio-community/Qwen3-VL-2B-Instruct-GGUF",
     "Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
     "VLM primary (page -> structured JSON)", 1080),
    ("vlm_mmproj",
     "lmstudio-community/Qwen3-VL-2B-Instruct-GGUF",
     "mmproj-Qwen3-VL-2B-Instruct-F16.gguf",
     "VLM projector (vision tower, F16)", 800),
    ("vlm_alt",
     "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
     "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
     "VLM alternate (slower, kept for A/B)", 1880),
    ("vlm_alt_mmproj",
     "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
     "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
     "VLM alternate projector (Q8_0)", 820),
    ("llm",
     "unsloth/granite-3.3-2b-instruct-GGUF",
     "granite-3.3-2b-instruct-Q4_K_M.gguf",
     "LLM primary (assembly / chat / KG)", 1400),
    ("llm_fallback",
     "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
     "qwen2.5-1.5b-instruct-q4_k_m.gguf",
     "LLM fallback for Tier-2 hardware", 1100),
    ("embed",
     "Mungert/all-MiniLM-L6-v2-GGUF",
     "all-MiniLM-L6-v2-q4_k_m.gguf",
     "Embedder (MiniLM-L6, 384-dim, Q4)", 25),
    ("embed_alt",
     "ggml-org/bge-small-en-v1.5-Q8_0-GGUF",
     "bge-small-en-v1.5-q8_0.gguf",
     "Embedder fallback (BGE-small, 384-dim)", 135),
]

RENAME = {"embed": "all-minilm-l6-v2-q4.gguf",
          "embed_alt": "bge-small-en-v1.5-q8_0.gguf"}


def target_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "models"


def download(key: str) -> Path:
    from huggingface_hub import hf_hub_download

    row = next(r for r in MANIFEST if r[0] == key)
    _, repo, filename, _role, _mb = row
    out = target_dir() / RENAME.get(key, filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[skip] {key}: {out.name} already present "
              f"({out.stat().st_size / 1e6:.0f} MB)")
        return out
    print(f"[get ] {key}: {repo}/{filename}")
    # huggingface_hub resumes partial downloads by default (no `resume=` kwarg)
    got = hf_hub_download(repo_id=repo, filename=filename,
                          local_dir=str(target_dir()))
    got = Path(got)
    if got != out:
        got.replace(out)
    print(f"[ok  ] {key}: {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the manifest")
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of manifest keys to fetch")
    args = ap.parse_args(argv)

    if args.list:
        for key, repo, fn, role, mb in MANIFEST:
            print(f"{key:12} {mb:5} MB  {role}\n"
                  f"             {repo}/{fn}")
        return 0

    keys = args.only or [r[0] for r in MANIFEST]
    failures = []
    for key in keys:
        try:
            download(key)
        except Exception as exc:  # keep going: fallbacks are optional
            failures.append((key, exc))
            print(f"[fail] {key}: {exc}", file=sys.stderr)
    if failures and args.only:
        return 1
    print("\nSummary:")
    for key, *_ in MANIFEST:
        p = target_dir() / RENAME.get(key, next(r[2] for r in MANIFEST if r[0] == key))
        state = f"{p.stat().st_size / 1e6:8.0f} MB" if p.exists() else "   MISSING"
        print(f"  {key:12} {state}  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
