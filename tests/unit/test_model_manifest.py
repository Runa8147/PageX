"""Model distribution: manifest, SHA-256 verify, download, offline pack.

Nothing here touches the network or a real 1 GB weight: the download path is
exercised with a fake ``huggingface_hub.hf_hub_download`` and verification
with a tiny manifest written to ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from pagex.core import model_manifest as mm

CONTENT = b"fake gguf payload for PageX tests\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def mini_manifest(tmp_path) -> Path:
    data = {
        "version": 1,
        "models": [
            {"key": "embed", "config_key": "embed_gguf",
             "role": "embedder (required)", "required": True, "tier": 1,
             "repo": "acme/embed-GGUF", "file": "embed.gguf",
             "dest": "embed.gguf", "size_bytes": len(CONTENT),
             "sha256": _sha(CONTENT), "license": "Apache-2.0"},
            {"key": "llm", "config_key": "llm_gguf",
             "role": "chat model (optional)", "required": False, "tier": 2,
             "repo": "acme/llm-GGUF", "file": "llm.gguf",
             "dest": "llm.gguf", "size_bytes": len(CONTENT),
             "sha256": _sha(CONTENT), "license": "Apache-2.0"},
        ],
    }
    path = tmp_path / "models_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the committed manifest (what users actually download)
# ---------------------------------------------------------------------------
def test_committed_manifest_is_complete_and_verifiable():
    entries = mm.load_manifest()
    keys = {e.key for e in entries}
    assert {"vlm", "vlm_mmproj", "llm", "embed"} <= keys
    required = {e.key for e in entries if e.required}
    assert required == {"vlm", "vlm_mmproj", "llm", "embed"}

    for e in entries:
        assert e.size_bytes > 0, e.key
        assert len(e.sha256) == 64 and e.sha256 == e.sha256.lower(), e.key
        assert e.dest and e.file and e.repo, e.key
        assert e.license in ("Apache-2.0", "MIT"), (e.key, e.license)
        assert e.download_url.startswith("https://huggingface.co/")
        # config_key is what pagex.toml / CFG.model_path() uses
        assert e.config_key.endswith("_gguf") or "mmproj" in e.config_key, e.key


def test_manifest_lives_inside_the_package():
    """The manifest must ship with wheels and frozen builds, not just git."""
    path = mm.manifest_path()
    assert path.exists()
    assert path.parent.name == "pagex"
    assert path.name == "models_manifest.json"


def test_select_by_key_tier_and_required():
    entries = mm.load_manifest()
    assert {e.key for e in mm.select(entries, ["vlm", "llm"])} == {"vlm", "llm"}
    tier2 = {e.key for e in mm.select(entries, tier=2)}
    assert {"vlm", "llm", "embed", "llm_fallback"} <= tier2
    assert "vlm_alt" not in tier2                # tier-3 alternates stay out
    with pytest.raises(KeyError):
        mm.select(entries, ["does-not-exist"])


# ---------------------------------------------------------------------------
# hashing + verify
# ---------------------------------------------------------------------------
def test_sha256_of_streams_the_file(tmp_path):
    p = tmp_path / "w.gguf"
    p.write_bytes(CONTENT)
    seen: list[tuple[str, int, int]] = []
    got = mm.sha256_of(p, progress=lambda k, d, t: seen.append((k, d, t)),
                       key="embed")
    assert got == _sha(CONTENT)
    assert seen and seen[-1][1] == seen[-1][2] == len(CONTENT)


def test_verify_statuses(tmp_path, mini_manifest):
    entries = mm.load_manifest(mini_manifest)
    dest = tmp_path / "models"
    dest.mkdir()

    # nothing installed yet
    report = {r["key"]: r for r in mm.verify(entries, dest_dir=dest)}
    assert report["embed"]["status"] == "missing"

    # correct bytes
    (dest / "embed.gguf").write_bytes(CONTENT)
    report = {r["key"]: r for r in mm.verify(entries, dest_dir=dest)}
    assert report["embed"]["status"] == "ok"
    assert report["embed"]["actual_sha256"] == _sha(CONTENT)
    assert report["llm"]["status"] == "missing"       # optional, still reported

    # wrong size
    (dest / "embed.gguf").write_bytes(CONTENT + b"!")
    report = {r["key"]: r for r in mm.verify(entries, dest_dir=dest)}
    assert report["embed"]["status"] == "size-mismatch"

    # right size, wrong bytes
    (dest / "embed.gguf").write_bytes(b"x" * len(CONTENT))
    report = {r["key"]: r for r in mm.verify(entries, dest_dir=dest)}
    assert report["embed"]["status"] == "sha-mismatch"

    # fast mode never hashes
    (dest / "embed.gguf").write_bytes(b"x" * len(CONTENT))
    report = {r["key"]: r for r in mm.verify(entries, dest_dir=dest,
                                             hash_files=False)}
    assert report["embed"]["status"] == "size-ok"


def test_missing_required_helpers(tmp_path, mini_manifest, monkeypatch):
    original = mm.load_manifest
    monkeypatch.setattr(mm, "load_manifest",
                        lambda *a, **kw: original(mini_manifest))
    dest = tmp_path / "models"
    dest.mkdir()
    assert mm.missing_required(dest) == ["embed"]
    (dest / "embed.gguf").write_bytes(CONTENT)
    assert mm.missing_required(dest) == []


# ---------------------------------------------------------------------------
# download (fake hub)
# ---------------------------------------------------------------------------
def test_download_fetches_verifies_and_is_idempotent(tmp_path, mini_manifest,
                                                     monkeypatch):
    import huggingface_hub as hub

    dest = tmp_path / "models"
    calls: list[str] = []

    def fake_hf_hub_download(repo_id, filename, local_dir=None, **kw):
        calls.append(f"{repo_id}/{filename}")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(CONTENT)
        return str(target)

    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)
    entries = mm.load_manifest(mini_manifest)

    first = mm.download(entries, dest_dir=dest)
    assert [r["status"] for r in first] == ["downloaded", "downloaded"]
    assert (dest / "embed.gguf").read_bytes() == CONTENT
    assert len(calls) == 2

    # second run skips what is already there (this is the resumable/cheap path)
    second = mm.download(entries, dest_dir=dest)
    assert [r["status"] for r in second] == ["present", "present"]
    assert len(calls) == 2

    # a corrupt download is detected and removed, not silently kept
    (dest / "embed.gguf").unlink()
    monkeypatch.setattr(
        hub, "hf_hub_download",
        lambda repo_id, filename, local_dir=None, **kw: _fake_write(
            Path(local_dir), filename, b"corrupted-bytes"))
    third = mm.download([entries[0]], dest_dir=dest)
    assert third[0]["status"] == "sha-mismatch"
    assert not (dest / "embed.gguf").exists()


def _fake_write(folder: Path, filename: str, payload: bytes) -> str:
    target = folder / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return str(target)


def test_download_error_is_reported_not_raised(tmp_path, mini_manifest,
                                               monkeypatch):
    import huggingface_hub as hub

    def boom(*a, **kw):
        raise OSError("no route to host")

    monkeypatch.setattr(hub, "hf_hub_download", boom)
    entries = mm.load_manifest(mini_manifest)
    report = mm.download(entries, dest_dir=tmp_path / "models")
    assert [r["status"] for r in report] == ["error", "error"]
    assert "no route to host" in report[0]["detail"]


# ---------------------------------------------------------------------------
# offline install pack (USB stick / air-gapped lab)
# ---------------------------------------------------------------------------
def test_install_pack_from_folder(tmp_path, mini_manifest):
    entries = mm.load_manifest(mini_manifest)
    dest = tmp_path / "models"
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "embed.gguf").write_bytes(CONTENT)
    (pack / "readme.txt").write_bytes(b"not a model")

    report = {r["file"]: r
              for r in mm.install_pack(pack, dest_dir=dest,
                                       manifest=mini_manifest)}
    assert report["embed.gguf"]["status"] == "installed"
    assert (dest / "embed.gguf").read_bytes() == CONTENT
    assert report["readme.txt"]["status"] == "ignored"
    # optional llm was not in the pack
    assert not (dest / "llm.gguf").exists()
    assert entries[0].dest == "embed.gguf"


def test_install_pack_rejects_corrupt_files(tmp_path, mini_manifest):
    dest = tmp_path / "models"
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "embed.gguf").write_bytes(b"tampered payload, wrong sha")

    report = {r["file"]: r
              for r in mm.install_pack(pack, dest_dir=dest,
                                       manifest=mini_manifest)}
    assert report["embed.gguf"]["status"] == "sha-mismatch"
    assert not (dest / "embed.gguf").exists(), "corrupt weight must not land"


def test_install_pack_from_zip(tmp_path, mini_manifest):
    dest = tmp_path / "models"
    src = tmp_path / "src"
    src.mkdir()
    (src / "embed.gguf").write_bytes(CONTENT)
    zip_path = tmp_path / "pagex-models-pack.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src / "embed.gguf", arcname="embed.gguf")

    report = {r["file"]: r
              for r in mm.install_pack(zip_path, dest_dir=dest,
                                       manifest=mini_manifest)}
    assert report["embed.gguf"]["status"] == "installed"
    assert (dest / "embed.gguf").read_bytes() == CONTENT
    # no leftovers from extraction
    assert not (dest / ".pack-tmp").exists()


def test_pack_models_roundtrip(tmp_path, mini_manifest, monkeypatch):
    original = mm.load_manifest
    monkeypatch.setattr(mm, "load_manifest",
                        lambda *a, **kw: original(mini_manifest))
    entries = original(mini_manifest)
    src = tmp_path / "models"
    src.mkdir()
    for e in entries:
        (src / e.dest).write_bytes(CONTENT)

    out = mm.pack_models(src, ["embed"], zip_path=tmp_path / "out.zip")
    assert out.exists()

    dest = tmp_path / "restored"
    report = {r["file"]: r
              for r in mm.install_pack(out, dest_dir=dest,
                                       manifest=mini_manifest)}
    assert report["embed.gguf"]["status"] == "installed"
    assert (dest / "embed.gguf").read_bytes() == CONTENT
