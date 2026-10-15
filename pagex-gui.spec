# -*- mode: python ; coding: utf-8 -*-
# ---------------------------------------------------------------------------
# PageX desktop app - PyInstaller **windowed** onedir build (name: pagex-gui)
#
# Build (always from the repository root):
#
#   Windows:  pyinstaller --noconfirm pagex-gui.spec
#             powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#   Linux:    .venv/bin/pyinstaller --noconfirm pagex-gui.spec
#             ./packaging/build_linux.sh
#
# Output:
#
#   Windows:  dist\pagex-gui\pagex-gui.exe   (no console window)
#   Linux:    dist/pagex-gui/pagex-gui
#
# Notes
# -----
# * collect_all() is mandatory for llama_cpp (native llama.cpp libs + vendor
#   payload), pymupdf (native MuPDF libs) and cv2 (opencv-headless native
#   lib): plain import tracing misses libraries loaded via ctypes or
#   ``__file__``-relative paths.
# * GGUF model weights are NEVER bundled (ADR-002 / docs/packaging.md).  The
#   weights are described by pagex/models_manifest.json and fetched at first
#   run with `pagex models download` (or installed offline with
#   `pagex models install-pack <zip>`).  The guard below fails the build if
#   any *.gguf or anything under a models/ directory slips into the bundle.
# * Entry point: pagex/gui/app.py uses relative imports (``from ..core...``).
#   PyInstaller executes the entry script as __main__ WITHOUT package
#   context, which breaks relative imports (pyinstaller issue #2560, still
#   open), so a 3-line stub containing only absolute imports is generated
#   outside the repository and frozen instead.
# ---------------------------------------------------------------------------

import atexit
import os
import shutil
import tempfile

from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = os.path.dirname(os.path.abspath(SPEC))

# ---------------------------------------------------------------------------
# entry stub (see header note about relative imports)
# ---------------------------------------------------------------------------
_stub_dir = tempfile.mkdtemp(prefix="pagex-gui-entry-")
atexit.register(shutil.rmtree, _stub_dir, ignore_errors=True)
entry = os.path.join(_stub_dir, "pagex_gui_entry.py")
with open(entry, "w", encoding="utf-8") as fh:
    fh.write(
        "import sys\n"
        "from pagex.gui.app import main\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n")

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

# pagex package data: pagex/models_manifest.json must ship so the frozen app
# can download / verify / install-pack its weights at runtime.
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
    name="pagex-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                              # windowed - no console (ADR-001)
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
    name="pagex-gui",
)
