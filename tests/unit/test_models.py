"""TC-UNIT-03 - ModelSlot: one resident heavy model, measurable RAM release.

The ADR-002 contract: the VLM and the LLM are *never* resident together and
``release()`` returns the RAM to the OS.  Skips cleanly when the GGUF files
are not downloaded; the assertions run on any machine that has them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pagex.core.config import CFG
from pagex.core.models import HEAVY, rss_gb

MODELS_OK = all(CFG.model_path(k).exists()
                for k in ("llm_gguf", "vlm_gguf", "vlm_mmproj"))

pytestmark = pytest.mark.skipif(
    not MODELS_OK, reason="GGUF models not downloaded (tools/download_models.py)")


@pytest.fixture(autouse=True)
def clean_slot():
    HEAVY.release()
    yield
    HEAVY.release()


def test_tc_unit_03_one_resident_model_at_a_time():
    """acquire() silently evicts whatever was loaded before (mutual exclusion)."""
    m_llm = HEAVY.acquire("llm", CFG.model_path("llm_gguf"), n_ctx=512)
    assert HEAVY.key == "llm" and HEAVY.loaded
    llm_only_rss = rss_gb()

    m_vlm = HEAVY.acquire("vlm", CFG.model_path("vlm_gguf"),
                          mmproj=CFG.model_path("vlm_mmproj"), n_ctx=512)
    assert HEAVY.key == "vlm"
    assert HEAVY.evictions >= 1
    # never both: the previous handle is invalidated by eviction
    assert m_llm is not m_vlm
    assert rss_gb() < llm_only_rss + 4.0          # sanity, not the budget


def test_tc_unit_03_release_returns_ram():
    """TC-UNIT-03 'wow' artifact: release drops RSS back near baseline."""
    baseline = rss_gb()
    HEAVY.acquire("llm", CFG.model_path("llm_gguf"), n_ctx=1024)
    loaded = rss_gb()
    assert loaded > baseline + 1.0, "model should be resident after acquire"
    HEAVY.release()
    after = rss_gb()
    assert not HEAVY.loaded
    assert after < loaded - 1.0, (
        f"release must give RAM back: {loaded:.2f} GB -> {after:.2f} GB")
    assert after < 2.0, f"resident after release too high: {after:.2f} GB"


def test_tc_unit_03_missing_model_error_is_actionable():
    with pytest.raises(FileNotFoundError) as exc:
        HEAVY.acquire("llm", Path("/nonexistent/model.gguf"), n_ctx=512)
    assert "download_models" in str(exc.value)
