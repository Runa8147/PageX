#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PageX Linux build: both PyInstaller specs + the portable tarball.
#
# Usage (from anywhere):   ./packaging/build_linux.sh
#
# Outputs (in the repository's dist/ folder):
#
#   dist/pagex-gui/pagex-gui          windowed desktop app (onedir)
#   dist/pagex/pagex                  console app (onedir)
#   dist/pagex-linux-x64.tar.gz       portable bundle: both apps + launcher
#
# Python is taken from PAGEX_BUILD_PYTHON, then .venv-build/, then .venv/,
# then python3 on PATH.  The `llm` extra (llama-cpp-python) is installed too:
# the frozen apps bundle the compiled llama.cpp library, so a compiler is
# required (build-essential + cmake on Debian/Ubuntu).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "repo: $ROOT"

# ---- pick a Python --------------------------------------------------------
pick_python() {
  if [ -n "${PAGEX_BUILD_PYTHON:-}" ]; then
    echo "$PAGEX_BUILD_PYTHON"
    return
  fi
  local cand
  for cand in "$ROOT/.venv-build/bin/python" "$ROOT/.venv/bin/python"; do
    if [ -x "$cand" ]; then echo "$cand"; return; fi
  done
  command -v python3 || true
}
PY="$(pick_python)"
if [ -z "$PY" ]; then
  echo "error: no Python found - install python3 or set PAGEX_BUILD_PYTHON" >&2
  exit 1
fi
echo "python: $PY"
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "error: Python >= 3.11 is required" >&2
  exit 1
fi

# ---- dependencies (pyinstaller ships in .[dev]; listed explicitly) ---------
"$PY" -m pip install -e ".[dev]" pyinstaller
# the `llm` extra = llama-cpp-python (compiles from source on first install)
"$PY" -m pip install -e ".[llm]"

# ---- PyInstaller: both specs ----------------------------------------------
for spec in pagex-gui.spec pagex-cli.spec; do
  echo ""
  echo "==> pyinstaller $spec"
  "$PY" -m PyInstaller --noconfirm "$spec"
done

# ---- verify the artifacts exist -------------------------------------------
GUI=dist/pagex-gui/pagex-gui
CLI=dist/pagex/pagex
for artifact in "$GUI" "$CLI"; do
  if [ ! -f "$artifact" ]; then
    echo "error: missing artifact: $artifact" >&2
    exit 1
  fi
done

# ---- portable tarball ------------------------------------------------------
# Layout: pagex-linux-x64/{pagex-gui,pagex,run-pagex.sh}.  The launcher keeps
# all state next to itself when a ./portable folder exists (touch -d it to
# opt in), otherwise the app falls back to ~/.local/share/pagex (platformdirs).
STAGE=dist/pagex-linux-x64
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -a dist/pagex-gui "$STAGE/pagex-gui"
cp -a dist/pagex     "$STAGE/pagex"
cat > "$STAGE/run-pagex.sh" <<'EOF'
#!/usr/bin/env bash
# PageX portable launcher.  If a ./portable folder sits next to this script,
# PAGEX_HOME points at it and ALL state (config, db, artifacts, models, logs)
# stays inside the extracted folder - otherwise state lives in
# ~/.local/share/pagex (Linux) / %LOCALAPPDATA%\pagex (Windows).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/portable" ]; then
  export PAGEX_HOME="$HERE/portable"
fi
exec "$HERE/pagex-gui/pagex-gui" "$@"
EOF
chmod 755 "$STAGE/run-pagex.sh"
tar -czf dist/pagex-linux-x64.tar.gz -C dist pagex-linux-x64

# ---- report ----------------------------------------------------------------
echo ""
echo "Qt runtime libraries on Debian/Ubuntu (if the GUI does not start):"
echo "    sudo apt install libegl1 libxcb-cursor0 libxkbcommon-x11-0"
echo ""
echo "artifacts:"
du -sh "$GUI" "$CLI" dist/pagex-linux-x64.tar.gz | sed 's/^/  /'
echo ""
ls -lh dist/pagex-linux-x64.tar.gz | sed 's/^/  /'
