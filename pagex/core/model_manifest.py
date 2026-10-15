"""Model distribution: manifest, download, SHA-256 verify, offline install pack.

ADR-002: every weight is a GGUF, and GGUFs are **never** bundled into frozen
binaries or wheels (they are 1-2 GB each).  Instead the repository ships a
small :data:`MANIFEST_NAME` describing each file (repo, filename, byte size,
SHA-256, license) and three operations:

===========================  ================================================
``pagex models download``    resumable fetch from Hugging Face -> data dir
``pagex models verify``      byte size + SHA-256 re-check of what is on disk
``pagex models install-pack`` offline install from a zip/USB stick (air-gapped)
===========================  ================================================

The manifest lives *inside the package* (``pagex/models_manifest.json``) so it
is present in editable installs, wheels and PyInstaller builds alike.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

log = logging.getLogger(__name__)

MANIFEST_NAME = "models_manifest.json"
_CHUNK = 1 << 20                      # 1 MiB streaming blocks

ProgressFn = Callable[[str, int, int], None]        # (key, done_bytes, total)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelEntry:
    key: str
    repo: str
    file: str
    dest: str
    role: str
    size_bytes: int
    sha256: str
    license: str = "Apache-2.0"
    config_key: str = ""
    required: bool = False
    tier: int = 1

    @property
    def download_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.file}"


def manifest_path() -> Path:
    here = Path(__file__).resolve().parents[1] / MANIFEST_NAME
    if here.exists():
        return here
    return Path(__file__).resolve().parent / MANIFEST_NAME   # pragma: no cover


def load_manifest(path: str | Path | None = None) -> list[ModelEntry]:
    """Parse the manifest (raises FileNotFoundError with a clear message)."""
    p = Path(path) if path else manifest_path()
    if not p.exists():
        raise FileNotFoundError(
            f"model manifest not found: {p} (expected inside the pagex package)")
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list[ModelEntry] = []
    for row in data.get("models", []):
        out.append(ModelEntry(
            key=row["key"], repo=row.get("repo", ""), file=row.get("file", ""),
            dest=row.get("dest") or row.get("file", ""),
            role=row.get("role", ""), size_bytes=int(row.get("size_bytes") or 0),
            sha256=str(row.get("sha256") or "").lower(),
            license=row.get("license", "Apache-2.0"),
            config_key=row.get("config_key", ""),
            required=bool(row.get("required", False)),
            tier=int(row.get("tier", 1))))
    if not out:
        raise ValueError(f"empty model manifest: {p}")
    return out


def select(entries: Sequence[ModelEntry], keys: Iterable[str] | None = None,
           *, tier: int | None = None, required_only: bool = False) -> list[ModelEntry]:
    """Filter the manifest by explicit keys and/or tier / required-ness."""
    out = list(entries)
    if keys:
        wanted = list(keys)
        known = {e.key for e in out}
        unknown = [k for k in wanted if k not in known]
        if unknown:
            raise KeyError(f"unknown model key(s): {', '.join(unknown)} "
                           f"(known: {', '.join(sorted(known))})")
        out = [e for e in out if e.key in set(wanted)]
    if required_only:
        out = [e for e in out if e.required]
    if tier is not None:
        out = [e for e in out if e.tier <= tier]
    return out


# ---------------------------------------------------------------------------
# hashing / verification
# ---------------------------------------------------------------------------
def sha256_of(path: str | Path, progress: ProgressFn | None = None,
              key: str = "") -> str:
    """Streaming SHA-256 (never loads a 2 GB weight into RAM)."""
    p = Path(path)
    total = p.stat().st_size
    h = hashlib.sha256()
    done = 0
    with open(p, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(key, done, total)
    return h.hexdigest()


def local_path(entry: ModelEntry, dest_dir: str | Path | None = None) -> Path:
    from .config import CFG
    base = Path(dest_dir) if dest_dir else CFG.models_dir
    return base / entry.dest


def verify(entries: Sequence[ModelEntry] | None = None, *,
           dest_dir: str | Path | None = None, hash_files: bool = True,
           progress: ProgressFn | None = None) -> list[dict]:
    """Check every selected weight on disk.

    Returns one dict per entry:
    ``{key, status, path, size_bytes, expected, sha256, detail}`` where
    status is ``ok | missing | size-mismatch | sha-mismatch | no-manifest-sha``.
    """
    entries = list(entries) if entries is not None else load_manifest()
    report: list[dict] = []
    for e in entries:
        p = local_path(e, dest_dir)
        row: dict = {"key": e.key, "path": str(p), "role": e.role,
                     "required": e.required, "expected": e.size_bytes,
                     "sha256": e.sha256 or None}
        if not p.exists():
            row.update(status="missing", size_bytes=0,
                       detail="run `pagex models download`")
            report.append(row)
            continue
        size = p.stat().st_size
        row["size_bytes"] = size
        if e.size_bytes and size != e.size_bytes:
            row.update(status="size-mismatch",
                       detail=f"expected {e.size_bytes} bytes, found {size}")
            report.append(row)
            continue
        if not e.sha256:
            row.update(status="no-manifest-sha", detail="size ok (no checksum)")
            report.append(row)
            continue
        if not hash_files:
            row.update(status="size-ok", detail="size ok (SHA-256 skipped)")
            report.append(row)
            continue
        got = sha256_of(p, progress=progress, key=e.key)
        ok = got.lower() == e.sha256.lower()
        row.update(status="ok" if ok else "sha-mismatch", actual_sha256=got,
                   detail="sha256 ok" if ok else "checksum mismatch - re-download")
        report.append(row)
    return report


# ---------------------------------------------------------------------------
# download (resumable, verified)
# ---------------------------------------------------------------------------
def download(entries: Sequence[ModelEntry] | None = None, *,
             dest_dir: str | Path | None = None,
             progress: ProgressFn | None = None,
             verify_after: bool = True) -> list[dict]:
    """Fetch *entries* into the models directory via ``huggingface_hub``.

    Partial downloads resume (huggingface_hub keeps them on disk), each file
    is moved into place atomically and then SHA-256 checked.
    """
    entries = list(entries) if entries is not None else load_manifest(required_only=True)
    dest = Path(dest_dir) if dest_dir else _default_dest()
    dest.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for e in entries:
        target = dest / e.dest
        row = {"key": e.key, "path": str(target), "role": e.role,
               "required": e.required}
        try:
            if target.exists() and (not e.size_bytes
                                    or target.stat().st_size == e.size_bytes):
                row["status"] = "present"
                row["detail"] = f"{target.name} already present"
                if progress:
                    progress(e.key, target.stat().st_size, target.stat().st_size)
                results.append(row)
                continue
            row.update(_fetch_one(e, dest, progress))
            if verify_after and e.sha256:
                got = sha256_of(target, progress=progress, key=e.key)
                if got.lower() != e.sha256.lower():
                    target.unlink(missing_ok=True)
                    row.update(status="sha-mismatch",
                               detail="downloaded file failed SHA-256; removed")
        except Exception as exc:                     # keep going: optional files
            log.warning("download %s failed: %s", e.key, exc)
            row.update(status="error", detail=str(exc))
        results.append(row)
    return results


def _default_dest() -> Path:
    from .config import CFG
    return CFG.models_dir


def _fetch_one(e: ModelEntry, dest: Path, progress: ProgressFn | None) -> dict:
    from huggingface_hub import hf_hub_download
    got = hf_hub_download(repo_id=e.repo, filename=e.file, local_dir=str(dest))
    got = Path(got)
    target = dest / e.dest
    if got != target:
        target.unlink(missing_ok=True)
        got.replace(target)
    if progress:
        size = target.stat().st_size
        progress(e.key, size, size)
    return {"status": "downloaded",
            "detail": f"{target.name} ({target.stat().st_size / 1e6:.0f} MB)"}


# ---------------------------------------------------------------------------
# offline install pack (USB / demo hall / air-gapped lab)
# ---------------------------------------------------------------------------
def install_pack(source: str | Path, *, dest_dir: str | Path | None = None,
                 verify_sha: bool = True,
                 manifest: str | Path | None = None) -> list[dict]:
    """Install weights from a **zip file or a folder** (no network).

    The pack may use either the manifest's ``dest`` or its original ``file``
    name; every file that matches a manifest entry is SHA-256 checked before
    it is put into the models directory.  *manifest* defaults to the manifest
    shipped inside the package (tests / mirrors may point at another one).
    """
    src = Path(source).expanduser()
    dest = Path(dest_dir) if dest_dir else _default_dest()
    dest.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(manifest)
    by_name: dict[str, ModelEntry] = {}
    for e in entries:
        by_name[e.dest.lower()] = e
        by_name[e.file.lower()] = e

    report: list[dict] = []
    if src.is_dir():
        files = [p for p in src.rglob("*") if p.is_file() and not p.name.startswith(".")]
        _install_files(files, by_name, dest, report, verify_sha)
    elif zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            names = [n for n in zf.namelist()
                     if not n.endswith("/") and not Path(n).name.startswith(".")]
            tmp = dest / ".pack-tmp"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                extracted = [Path(zf.extract(n, tmp)) for n in names]
                _install_files(extracted, by_name, dest, report, verify_sha)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    else:
        raise FileNotFoundError(f"not a zip file or directory: {src}")
    return report


def _install_files(files: Sequence[Path], by_name: dict[str, ModelEntry],
                   dest: Path, report: list[dict], verify_sha: bool) -> None:
    for f in files:
        entry = by_name.get(f.name.lower())
        row: dict = {"file": f.name, "path": str(f), "size_bytes": f.stat().st_size}
        if entry is None:
            row.update(status="ignored", detail="not in the model manifest")
            report.append(row)
            continue
        if verify_sha and entry.sha256:
            got = sha256_of(f)
            if got.lower() != entry.sha256.lower():
                row.update(key=entry.key, status="sha-mismatch",
                           detail="pack file corrupt - refusing to install")
                report.append(row)
                continue
        target = dest / entry.dest
        if target.exists() and target.stat().st_size == f.stat().st_size \
                and f.resolve() != target.resolve():
            row.update(key=entry.key, status="present",
                       detail=f"{target.name} already installed")
            report.append(row)
            continue
        if f.resolve() != target.resolve():
            shutil.copy2(f, target)
        row.update(key=entry.key, status="installed",
                   detail=f"-> {target}")
        report.append(row)


# ---------------------------------------------------------------------------
# small helpers for the CLI / GUI wizard
# ---------------------------------------------------------------------------
def status_summary(report: Sequence[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in report:
        counts[row.get("status", "?")] = counts.get(row.get("status", "?"), 0) + 1
    return counts


def missing_required(dest_dir: str | Path | None = None) -> list[str]:
    entries = select(load_manifest(), required_only=True)
    return [e.key for e in entries if not local_path(e, dest_dir).exists()]


def pack_models(dest_dir: str | Path, keys: Iterable[str] | None = None,
                *, zip_path: str | Path | None = None) -> Path:
    """Build an offline demo pack (zip) from the weights already on disk."""
    src = Path(dest_dir)
    entries = select(load_manifest(), keys)
    files = [local_path(e, src) for e in entries]
    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "cannot build a pack, missing: " + ", ".join(p.name for p in missing))
    out = Path(zip_path) if zip_path else src / "pagex-models-pack.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    return out
