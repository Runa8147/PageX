#!/usr/bin/env python3
"""Build the PageX **Offline Demo Pack** zip (zero-install, USB / air-gapped).

    python packaging/make_demo_pack.py --app dist/pagex-gui
    python packaging/make_demo_pack.py --app dist/pagex-gui --models models \
        --out dist/pagex-demo-pack-with-models.zip

Layout of the produced zip:

    README-DEMO.txt          5-step instructions
    Run-PageX.bat            Windows launcher (PAGEX_HOME=<pack>\\page-data)
    run-pagex.sh             Linux launcher    (same, mode 755)
    app/                     the frozen onedir build (--app)
    samples/                 the sample corpus (--samples, default samples/)
    models/                  GGUF weights - ONLY with --models (~7 GB!)
    models_manifest.json     copy of pagex/models_manifest.json - ONLY with --models

Weights are OFF by default: the pack is meant to stay small; machines
without internet install the models once from any copy with
`pagex models install-pack <zip-or-folder>` (ADR-002, docs/packaging.md).

Pure stdlib (argparse / pathlib / zipfile). `.gguf` members are written
with ZIP_STORED: deflate gains ~0 % on already-quantised weights and only
burns CPU on multi-GB files.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "pagex" / "models_manifest.json"

#: never copy caches/OS junk from the frozen onedir into the pack
SKIP_NAMES = {"__pycache__", ".DS_Store", "Thumbs.db"}
SKIP_SUFFIX = {".pyc", ".pyo"}

LAUNCHER_BAT = "\r\n".join([
    "@echo off",
    "rem PageX offline demo launcher - all state stays in .\\page-data",
    'set "PAGEX_HOME=%~dp0page-data"',
    'start "" "%~dp0app\\pagex-gui.exe"',
    "",
])

LAUNCHER_SH = """\
#!/usr/bin/env bash
# PageX offline demo launcher - all state stays in ./page-data
export PAGEX_HOME="$(dirname "$0")/page-data"
exec "$(dirname "$0")/app/pagex-gui" "$@"
"""


# ---------------------------------------------------------------------------
# README-DEMO.txt (exactly 5 steps)
# ---------------------------------------------------------------------------
def _cli_block(cli_rel: str | None, action: str, indent: str = "") -> str:
    """Terminal snippet for `pagex <action>` - use the pack's own CLI if it
    was bundled, otherwise point at a CLI that actually exists."""
    if cli_rel:
        win = cli_rel.replace("/", "\\")
        body = (
            "      cmd (Windows, run from this folder):\n"
            '          set "PAGEX_HOME=%CD%\\page-data"\n'
            f"          {win} {action}\n"
            "      bash (Linux):\n"
            '          export PAGEX_HOME="$PWD/page-data"\n'
            f"          ./{cli_rel} {action}\n")
    else:
        body = (
            "      This pack contains only the GUI - run the same command with any\n"
            "      PageX command line you have (a source checkout's `python cli.py`,\n"
            "      the pagex-linux-x64 tarball's CLI, ...), from THIS pack's folder:\n"
            f'          PAGEX_HOME="$PWD/page-data" <pagex-cli> {action}\n'
            "      or rebuild the pack with the CLI included (packaging/README.md).\n")
    if not indent:
        return body
    return "".join(indent + line for line in body.splitlines(keepends=True))


def _readme(include_models: bool, cli_rel: str | None) -> str:
    if include_models:
        step2 = (
            "    The weights ALREADY ship in this pack under .\\models.\n"
            "    Install them once - offline, SHA-256 verified against the\n"
            "    bundled models_manifest.json:\n"
            + _cli_block(cli_rel, "models install-pack models"))
    else:
        step2 = (
            "    Model weights are NOT included (that is what keeps this zip\n"
            "    small). Pick ONE option:\n"
            "      a) online: fetch the ~7 GB of GGUFs once - resumable and\n"
            "         SHA-256 verified:\n"
            + _cli_block(cli_rel, "models download", indent="  ")
            + "      b) offline: copy pagex-models-pack.zip (or a models folder)\n"
            "         from a machine that already has the weights, then:\n"
            + _cli_block(cli_rel, "models install-pack <zip-or-folder>",
                         indent="  "))

    if cli_rel:
        step4 = (
            "    The same core is available from a terminal in this folder:\n"
            '      cmd:  set "PAGEX_HOME=%CD%\\page-data"  &  '
            f"{cli_rel.replace('/', chr(92))} search \"moment of inertia\"\n"
            '      bash: export PAGEX_HOME="$PWD/page-data" &  '
            f"./{cli_rel} search \"moment of inertia\"\n")
    else:
        step4 = (
            "    (This pack contains only the GUI - the Search and Ask tabs do\n"
            "     everything the `pagex` command line does.)\n")

    return f"""\
PageX - Offline Demo Pack
=========================
Everything runs on THIS machine: no cloud API, no account, no telemetry.
All state lives in the page-data folder created by the launchers.

Step 1 - EXTRACT
    Unzip this whole folder anywhere (USB stick, Desktop, D:\\...).
    Keep the folder structure - the launchers use paths relative to
    themselves.

Step 2 - GET THE MODEL WEIGHTS (~7 GB, once)
{step2}
Step 3 - START THE APP
    Windows: double-click  Run-PageX.bat
    Linux:   ./run-pagex.sh            (chmod +x run-pagex.sh if needed)
    Both launchers set PAGEX_HOME to the page-data folder next to them, so
    config, database, artifacts, models and logs stay inside this pack.

Step 4 - INGEST AND ASK
    In the app press Ingest and pick a photo, scan or PDF - or use the
    bundled corpus in samples\\ (queries.json lists ready-made queries).
    Then search or ask for grounded answers with page citations.
{step4}
Step 5 - BACK UP / RESET
    Copy this entire folder to keep everything (the weights are the big
    part). Deleting page-data resets PageX to factory state - nothing is
    ever written outside this pack folder.
"""


# ---------------------------------------------------------------------------
# zip helpers
# ---------------------------------------------------------------------------
def _add_text(zf: zipfile.ZipFile, arcname: str, text: str,
              mode: int = 0o644) -> None:
    info = zipfile.ZipInfo(arcname)               # deterministic 1980 stamp
    info.create_system = 3                        # unix - keeps the exec bit
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, text)


def _add_tree(zf: zipfile.ZipFile, src: Path, prefix: str,
              skip_suffix: frozenset[str] = frozenset()) -> tuple[int, int]:
    """Add *src* recursively under *prefix*; returns (files, bytes)."""
    added = nbytes = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if any(part in SKIP_NAMES for part in rel.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIX or path.suffix.lower() in skip_suffix:
            continue
        # arcnames stay shallow (app/..., models/...) and always use "/" so
        # extraction is safe even on Windows path-length-limited shells.
        stored = path.suffix.lower() == ".gguf"
        zf.write(path, f"{prefix}/{rel.as_posix()}",
                 compress_type=(zipfile.ZIP_STORED if stored
                                else zipfile.ZIP_DEFLATED))
        added += 1
        nbytes += path.stat().st_size
    return added, nbytes


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="build the PageX offline demo pack (zip)")
    ap.add_argument("--app", type=Path, required=True,
                    help="frozen onedir to include, e.g. dist/pagex-gui")
    ap.add_argument("--models", type=Path, default=None,
                    help="folder of GGUF weights to include "
                         "(default: OFF - makes a ~7 GB zip)")
    ap.add_argument("--samples", type=Path, default=Path("samples"),
                    help="sample corpus to include (default: samples/)")
    ap.add_argument("--out", type=Path, default=Path("dist/pagex-demo-pack.zip"),
                    help="output zip (default: dist/pagex-demo-pack.zip)")
    args = ap.parse_args(argv)

    app: Path = args.app
    if not app.is_dir():
        sys.exit(f"error: --app directory not found: {app}")
    if not ((app / "pagex-gui.exe").is_file() or (app / "pagex-gui").is_file()):
        sys.exit(f"error: {app} contains no pagex-gui(.exe) - pass the frozen "
                 "onedir built by pagex-gui.spec (e.g. dist/pagex-gui)")
    samples: Path = args.samples
    if not samples.is_dir():
        sys.exit(f"error: --samples directory not found: {samples}")

    models: Path | None = args.models
    if models is not None:
        if not models.is_dir():
            sys.exit(f"error: --models directory not found: {models}")
        if not MANIFEST.is_file():
            sys.exit(f"error: model manifest not found: {MANIFEST}")
        if not any(models.rglob("*.gguf")):
            print(f"warning: no *.gguf under {models} - packing anyway")

    # does the onedir contain the console app too? (packs are usually GUI-only)
    cli_rel = "app/pagex" if ((app / "pagex.exe").is_file()
                              or (app / "pagex").is_file()) else None

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    readme = _readme(models is not None, cli_rel)
    print(f"building {out} ...")
    with zipfile.ZipFile(out, "w") as zf:
        _add_text(zf, "README-DEMO.txt", readme)
        _add_text(zf, "Run-PageX.bat", LAUNCHER_BAT)
        _add_text(zf, "run-pagex.sh", LAUNCHER_SH, mode=0o755)

        n, b = _add_tree(zf, app, "app")
        print(f"  app/                 {n:5d} files  {b / 1e6:9.1f} MB")
        n, b = _add_tree(zf, samples, "samples")
        print(f"  samples/             {n:5d} files  {b / 1e6:9.1f} MB")
        if models is not None:
            zf.write(MANIFEST, "models_manifest.json",
                     compress_type=zipfile.ZIP_DEFLATED)
            # skip *.zip inside models/ - `pagex models pack` drops its own
            # archive there and re-packing it would double the size.
            n, b = _add_tree(zf, models, "models",
                             skip_suffix=frozenset({".zip"}))
            print(f"  models/              {n:5d} files  {b / 1e6:9.1f} MB")

    size = out.stat().st_size
    print(f"wrote {out} ({size / 1e6:.1f} MB)")
    if models is None:
        print("tip: add --models <dir> to bundle the weights (~7 GB zip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
