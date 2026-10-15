# PageX — Packaging & Distribution

How a commit becomes something a lecturer's Windows laptop, an air-gapped lab
machine or a headless server can run — and where every artifact comes from.
Companion files: `packaging/README.md` (command index), `NOTICE` (licenses),
ADR-001/002 for the constraints behind the choices.

---

## 1. The distribution ladder

| Tier | Artifact | Who uses it | Effort to build |
|---|---|---|---|
| 0 | git checkout + `pip install -e ".[dev]"` (add `.[llm]` for inference) | developers, contributors, CI | low — needs Python ≥ 3.11, `llm` needs a compiler |
| 1 | frozen binaries: `PageX-Setup-x64.exe` (Windows), `pagex-linux-x64.tar.gz` (Linux) | demo machines, lecturers, users with no pip | medium — PyInstaller onedir per OS |
| 2 | offline demo pack: `pagex-demo-pack.zip` | air-gapped labs, USB demos | low, once Tier 1 works (`make_demo_pack.py`) |
| 3 | Docker image `pagex:cpu` | servers, CI, headless automation | low — `docker build`; CLI only |
| 4 | PyPI wheel + GitHub Releases | `pip install pagex`, general public | GitHub Releases automated on `v*` tags; PyPI publish deferred (§7) |

One code base, one `pagex.toml`, five entry ramps: Tier 0 assumes a
developer machine, Tier 1 assumes nothing, Tier 2 assumes no network at all,
Tier 3 assumes Docker and gives up the GUI, Tier 4 is the packaging-of-record.

## 2. Why frozen binaries are the primary distribution

* The one dependency a normal user machine cannot satisfy is
  `llama-cpp-python`: it compiles llama.cpp from source. Windows needs the
  Visual Studio Build Tools (~2 GB) + CMake, Linux `build-essential` + cmake,
  macOS the Xcode command-line tools. A demo laptop has none of them, and a
  plain `pip install pagex` would fail at exactly that package (it ships an
  sdist, not a wheel).
* Everything else — Python 3.11, PyQt6/Qt, MuPDF, OpenCV, numpy — is bundled
  into the onedir by PyInstaller, so the target machine needs no runtime,
  no venv and no shell. Double-click `pagex-gui.exe`, or `./run-pagex.sh`.
* Consequence: the `llm` extra in `pyproject.toml` keeps *pip* installs
  compiler-free (`.[dev]` / a bare wheel never pulls llama-cpp), while the
  frozen binaries — built by `packaging/build_*` — install `.[llm]` and
  bundle the compiled library. Tier 0 stays for development, Tier 1 for use.
* Docker has the same compiler problem one level down; the `Dockerfile`
  installs `build-essential` + `cmake` in the image itself (or can use a
  prebuilt CPU wheel — see the comment in the file).

## 3. Model distribution

Every weight is a GGUF described by `pagex/models_manifest.json`, which ships
*inside* the package (editable install, wheel and frozen build alike):
repository, filename, byte size, SHA-256, license, tier, required-flag.

| command | what it does |
|---|---|
| `pagex models download` | resumable fetch (huggingface_hub) into the data dir, SHA-256 verified per file |
| `pagex models verify` | re-check byte size + SHA-256 of whatever is on disk |
| `pagex models install-pack <zip\|folder>` | offline install from USB / another machine, refuses corrupt files |
| `pagex models pack` | build such a zip from weights already on disk |

**Why weights never ship inside binaries, wheels or the default docker
layer** (ADR-002):

* they are 1.1–1.9 GB each (~7 GB total): a binary would dwarf the app, an
  image would not fit a demo-hall network, a release would take hours to
  download;
* one copy per machine in the data directory instead of one copy per
  artifact — download once, use from GUI, CLI and docker alike;
* quantisation choices (Q4_K_M primary vs. fallbacks) change more often than
  code; bundled weights would force a re-release per switch;
* it is enforced, not just documented: `pagex-gui.spec` / `pagex-cli.spec`
  assert that no `*.gguf` or `models/` path entered the bundle, and
  `.dockerignore` keeps weights out of the build context.

**Licenses** (full detail in `NOTICE`):

| component | license |
|---|---|
| PageX code | MIT (`LICENSE`, © 2026 PageX Team) |
| Qwen3-VL-2B, Qwen2.5-VL-3B, granite-3.3-2b, Qwen2.5-1.5B, all-MiniLM-L6-v2 (weights) | Apache-2.0 |
| bge-small-en-v1.5 (weights) | MIT |
| frozen-build runtime libs (PyQt6/Qt, PyMuPDF, llama-cpp-python, OpenCV, ReportLab, …) | see `NOTICE` — PyMuPDF is AGPL-3.0-or-commercial |

`NOTICE` must travel with any redistribution of the weights (Apache-2.0
attribution); `packaging/make_demo_pack.py` therefore writes a copy of the
manifest next to the models it packs.

## 4. Portable mode (`PAGEX_HOME`) and where state lives

`pagex.core.paths` answers every "where does X live" question through one
ladder, highest priority first:

| # | mode | when | config | writable state (db, artifacts, models, logs) |
|---|---|---|---|---|
| 1 | portable | `PAGEX_HOME` is set | `$PAGEX_HOME/pagex.toml` | `$PAGEX_HOME/` — the whole app state in one folder |
| 2 | source checkout | running from the git repo | `./pagex.toml` | `./data`, `./models` |
| 3 | frozen | `sys.frozen` (PyInstaller), no `PAGEX_HOME` | platformdirs config dir | platformdirs data dir |
| 4 | installed | `pip install pagex` | platformdirs config dir | platformdirs data dir |

platformdirs locations for modes 3 and 4:

| OS | config | data |
|---|---|---|
| Windows | `%APPDATA%\pagex\pagex.toml` | `%LOCALAPPDATA%\pagex\` |
| Linux | `~/.config/pagex/pagex.toml` | `~/.local/share/pagex/` |

Everything shipped sets mode 1 deliberately: the demo-pack launchers export
`PAGEX_HOME` to `<pack>/page-data`, the Linux tarball launcher switches to
`./portable` when that folder exists, and the docker image runs with
`ENV PAGEX_HOME=/data`. Deleting that folder is a factory reset; copying it
is a full backup/migration.

## 5. Packaging files reference

| file | what it does |
|---|---|
| `pagex-gui.spec` | windowed PyInstaller onedir for the PyQt6 desktop app; `collect_all` for `llama_cpp`/`pymupdf`/`cv2`, ships `pagex.toml` + `samples/` + `models_manifest.json`, asserts no GGUF enters the bundle |
| `pagex-cli.spec` | console PyInstaller onedir (`pagex`) built from the root `cli.py` shim; same collection and GGUF guard |
| `Dockerfile` | headless CLI image `pagex:cpu` on `python:3.11-slim-bookworm`; installs `/app[llm]`, verifies imports, `PAGEX_HOME=/data` |
| `.dockerignore` | keeps `models/`, `data/`, git history, venvs, build outputs and the large scanned PDF out of the build context |
| `.github/workflows/ci.yml` | `test` matrix (2 OS × 2 Python) on every push/PR; tag-gated `package` job that builds and publishes releases |
| `packaging/windows/pagex.iss` | Inno Setup 6 script: per-user `PageX-Setup-x64.exe` from `dist\pagex-gui\` |
| `packaging/build_windows.ps1` | Windows driver: Python check, deps (`.[dev]`, `.[llm]`, pyinstaller), both specs, artifact verification, optional ISCC, size report |
| `packaging/build_linux.sh` | Linux driver: same steps + portable tarball with `run-pagex.sh` launcher and Qt apt hint |
| `packaging/make_demo_pack.py` | builds the offline demo pack zip (app + samples + launchers; `--models` adds weights) |
| `packaging/README.md` | one-page index of artifacts and exact build commands |
| `docs/packaging.md` | this document |
| `NOTICE` / `LICENSE` | third-party attribution / MIT license |

Build commands (both build drivers may be run from any directory):

```bash
# Linux                                          # Windows (repo root)
./packaging/build_linux.sh                       powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

# either spec directly (from the repository root):
.venv/bin/pyinstaller --noconfirm pagex-gui.spec    # Linux
pyinstaller --noconfirm pagex-cli.spec              # Windows
```

## 6. CI matrix

| job | trigger | matrix | what it proves |
|---|---|---|---|
| `test` | every push / PR | ubuntu-latest, windows-latest × Python 3.11, 3.12 | the suite passes on both OSes with `QT_QPA_PLATFORM=offscreen` and no compiler (`.[dev]` only, `-m "not slow"`) |
| `package` | tags `v*`, after `test` | ubuntu-latest, windows-latest × Python 3.11 | the real release artifacts build: tarball on Linux, onedir + Inno installer on Windows |

OS parity is the point of the matrix: Qt platform plugins, path handling,
SQLite/locking behaviour, PyMuPDF wheels and PyInstaller hooks all differ
per OS, so every commit — not only release day — is proven on both. The
`package` job uploads the artifacts and attaches them to the GitHub Release
(`softprops/action-gh-release`, `dist/*.tar.gz` + `dist/*.exe`).

A pushed tag therefore produces: `PageX-Setup-x64.exe` (Windows) and
`pagex-linux-x64.tar.gz` (Linux) as release assets — Tier 1 in §1.

## 7. Deliberately deferred

* **PyPI publish.** The wheel is buildable today (`python -m build`) and the
  `llm` extra keeps installs compiler-free, but trusted publishing, version
  automation and the `[llm]` install story on PyPI are deferred; GitHub
  Releases already cover Tier 1/Tier 4 binary needs.
* **AppImage / .deb.** The Linux tarball covers Tier 1 with zero packaging
  infrastructure; AppImage (FUSE) and .deb builds wait for an actual Linux
  packaging request.
* **QtWebEngine — stays out (ADR-001).** The GUI is QtWidgets only; there is
  no web engine in any artifact to package, ~300 MB are saved per install,
  and no second event loop exists to maintain. Any future rich-text view
  remains widget-based rather than pulling QtWebEngine into the bundle.
