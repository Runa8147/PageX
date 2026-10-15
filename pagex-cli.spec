# -*- mode: python ; coding: utf-8 -*-
# ---------------------------------------------------------------------------
# PageX command line app - PyInstaller **console** onedir build (name: pagex)
#
# Build (always from the repository root):
#
#   Windows:  pyinstaller --noconfirm pagex-cli.spec
#             powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#   Linux:    .venv/bin/pyinstaller --noconfirm pagex-cli.spec
#             ./packaging/build_linux.sh
#
# Output:
#
#   Windows:  dist\pagex\pagex.exe
#   Linux:    dist/pagex/pagex
#
# Entry script: the repository-root ``cli.py`` shim
# (``from pagex.cli import main`` + ``if __name__ == "__main__"``).  It is a
# plain top-level script with absolute imports only, which is exactly what
# PyInstaller needs; the real CLI lives in ``pagex/cli.py`` (also the
# ``pagex`` console script from pyproject.toml).
#
# Same rules as pagex-gui.spec:
# * collect_all() for llama_cpp / pymupdf / cv2 (native libs are not found by
#   import tracing alone),
# * collect_data_files("pagex") so pagex/models_manifest.json ships,
# * GGUF model weights are NEVER bundled (ADR-002) - the guard at the bottom
#   fails the build if a *.gguf or anything under models/ sneaks in.
# ---------------------------------------------------------------------------

import os

from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = os.path.dirname(os.path.abspath(SPEC))

entry = os.path.join(ROOT, "cli.py")
with open(entry, encoding="utf-8") as fh:
    _entry_src = fh.read()
assert "if __name__" in _entry_src, (
    "cli.py must keep its `if __name__ == '__main__'` guard - it is the "
    "PyInstaller entry script of the frozen `pagex` console app")

# ---------------------------------------------------------------------------
# native libraries, package data, config + samples
# ---------------------------------------------------------------------------
datas = []
binaries = []
hiddenimports = []
for pkg in ("llama_cpp", "pymupdf", "cv2"):       # see header note
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# pagex package data: pagex/models_manifest.json must ship so `pagex models
# download|verify|install-pack` works in the frozen build.
datas += collect_data_files("pagex")

datas += [
    (os.path.join(ROOT, "pagex.toml"), "."),      # default config (read-only)
    (os.path.join(ROOT, "samples"), "samples"),   # sample corpus (`pagex demo`)
]


# ---------------------------------------------------------------------------
# ADR-002 guard - model weights never ship inside binaries
# ---------------------------------------------------------------------------
def _is_weight(toc_entry) -> bool:
    """True if a TOC entry (dest, src, typecode) refers to model weights."""
    for part in toc_entry[:2]:
        path = str(part).replace("\\", "/").lower()
        if path.endswith(".gguf"):
            return True
        if "models" in path.split("/"):           # a models/ directory segment
            return True
    return False


datas = [t for t in datas if not _is_weight(t)]
binaries = [t for t in binaries if not _is_weight(t)]

a = Analysis(
    [entry],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Post-Analysis sweep: drop anything Analysis pulled in on its own, then fail
# loudly if a weight still survived - never ship GGUFs (ADR-002).
a.datas = [t for t in a.datas if not _is_weight(t)]
a.binaries = [t for t in a.binaries if not _is_weight(t)]
leaked = [t for t in list(a.scripts) + list(a.binaries) + list(a.datas)
          if _is_weight(t)]
assert not leaked, f"model weights must never be bundled: {leaked}"

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pagex",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,                               # console app: stdout/exit codes
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pagex",
)
