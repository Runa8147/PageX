# PageX packaging - command index

Everything needed to turn this repository into a distributable artifact.
Design and rationale: [`docs/packaging.md`](../docs/packaging.md).
Model licenses: [`NOTICE`](../NOTICE). **No artifact ever contains GGUF
weights** (ADR-002) - they are fetched with `pagex models download`.

## Files in this folder

| file | artifact | kind |
|---|---|---|
| `../pagex-gui.spec` | `dist/pagex-gui/` | PyInstaller spec, windowed desktop app |
| `../pagex-cli.spec` | `dist/pagex/` | PyInstaller spec, console app |
| `build_windows.ps1` | both onedirs + `PageX-Setup-x64.exe` | Windows build driver |
| `build_linux.sh` | both onedirs + `pagex-linux-x64.tar.gz` | Linux build driver |
| `windows/pagex.iss` | `dist/PageX-Setup-x64.exe` | Inno Setup 6 installer script |
| `make_demo_pack.py` | `dist/pagex-demo-pack.zip` | offline demo pack (zip) |
| `../Dockerfile` | `pagex:cpu` | headless CLI image |
| `../.github/workflows/ci.yml` | GitHub Release assets on `v*` tags | CI |

## Build commands

### Windows (Python 3.11+, run from the repository root)

```powershell
# both onedirs + installer (installs pyinstaller + the `llm` extra first;
# uses .venv-build\ if present, otherwise the python on PATH)
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

# or by hand:
python -m pip install -e ".[dev]" pyinstaller -e ".[llm]"
pyinstaller --noconfirm pagex-gui.spec
pyinstaller --noconfirm pagex-cli.spec
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\pagex.iss
```

Outputs: `dist\pagex-gui\pagex-gui.exe`, `dist\pagex\pagex.exe`,
`dist\PageX-Setup-x64.exe` (per-user install, no admin rights).

### Linux (Python 3.11+, run from anywhere)

```bash
./packaging/build_linux.sh          # chmod +x is already set
```

Outputs: `dist/pagex-gui/pagex-gui`, `dist/pagex/pagex`,
`dist/pagex-linux-x64.tar.gz` (extracts to `pagex-linux-x64/` with a
`run-pagex.sh` launcher). Qt runtime libs on Debian/Ubuntu:

```bash
sudo apt install libegl1 libxcb-cursor0 libxkbcommon-x11-0
```

### Offline demo pack (either OS, after a GUI build)

```bash
python packaging/make_demo_pack.py --app dist/pagex-gui
# with weights (~7 GB):  --models models
```

Output: `dist/pagex-demo-pack.zip` (app + samples + launchers + README).

### Docker (headless CLI)

```bash
docker build -t pagex:cpu .
docker run --rm -v ~/docs:/docs -v pagex-data:/data pagex:cpu ingest /docs/x.pdf
```

State lives in the `/data` volume (`PAGEX_HOME=/data`); weights are added
once with `docker run --rm -v pagex-data:/data pagex:cpu models download`.

## Releases

Pushing a tag `v*` runs `.github/workflows/ci.yml`: the `test` matrix
(Ubuntu + Windows, Python 3.11 + 3.12 + 3.13) must pass, then the `package` job
builds the artifacts above and attaches them to the GitHub Release.
