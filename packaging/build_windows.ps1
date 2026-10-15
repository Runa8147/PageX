<#
.SYNOPSIS
  Build every Windows distribution artifact for PageX.

  Runs from anywhere (repo root or packaging/); all output lands in the
  repository's dist\ folder:

    dist\pagex-gui\pagex-gui.exe   windowed desktop app (onedir, pagex-gui.spec)
    dist\pagex\pagex.exe           console app (onedir, pagex-cli.spec)
    dist\PageX-Setup-x64.exe       Inno Setup installer (when ISCC.exe found)

  Requirements:
    * Python 3.11+ - taken from PAGEX_BUILD_PYTHON, then .venv-build\, then PATH
      (to isolate the build env:  python -m venv .venv-build  - it is only
      used if it already exists; nothing is created behind your back)
    * network for the first `pip install` (pyinstaller + the `llm` extra =
      llama-cpp-python, which compiles llama.cpp - VS Build Tools required)
    * optionally Inno Setup 6 for the installer, e.g.  choco install innosetup
#>
[CmdletBinding()]
param(
    [switch]$NoInstaller       # skip the Inno Setup step even if ISCC.exe exists
)
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    Write-Host "repo: $RepoRoot"

    # ---- pick a Python --------------------------------------------------
    $python = $env:PAGEX_BUILD_PYTHON
    if (-not $python) {
        $venvPy = Join-Path $RepoRoot ".venv-build\Scripts\python.exe"
        if (Test-Path $venvPy) { $python = $venvPy }
    }
    if (-not $python) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $python = $cmd.Source }
    }
    if (-not $python) {
        throw "no Python found - install Python 3.11+ or set PAGEX_BUILD_PYTHON"
    }
    Write-Host "python: $python"
    & $python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Python >= 3.11 is required" }

    # ---- dependencies (pyinstaller ships in .[dev]; listed explicitly) ----
    & $python -m pip install -e ".[dev]" pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pip install -e `".[dev]`" pyinstaller failed" }
    # the `llm` extra = llama-cpp-python: the frozen apps must bundle the
    # compiled llama.cpp library (first run compiles from source, ~10 min).
    & $python -m pip install -e ".[llm]"
    if ($LASTEXITCODE -ne 0) { throw "pip install -e `".[llm]`" failed (VS Build Tools required)" }

    # ---- PyInstaller: both specs ----------------------------------------
    foreach ($spec in "pagex-gui.spec", "pagex-cli.spec") {
        Write-Host ""
        Write-Host "==> pyinstaller $spec"
        & $python -m PyInstaller --noconfirm $spec
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed: $spec" }
    }

    # ---- verify the artifacts exist -------------------------------------
    $gui = Join-Path $RepoRoot "dist\pagex-gui\pagex-gui.exe"
    $cli = Join-Path $RepoRoot "dist\pagex\pagex.exe"
    foreach ($artifact in @($gui, $cli)) {
        if (-not (Test-Path $artifact)) { throw "missing artifact: $artifact" }
    }

    # ---- Inno Setup installer (optional) --------------------------------
    $setup = Join-Path $RepoRoot "dist\PageX-Setup-x64.exe"
    if ($NoInstaller) {
        Write-Host ""
        Write-Host "installer skipped (-NoInstaller)"
    } else {
        $iscc = $null
        $cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
        if ($cmd) { $iscc = $cmd.Source }
        if (-not $iscc) {
            foreach ($base in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
                if ($base) {
                    $cand = Join-Path $base "Inno Setup 6\ISCC.exe"
                    if (Test-Path $cand) { $iscc = $cand; break }
                }
            }
        }
        if ($iscc) {
            Write-Host ""
            Write-Host "==> ISCC packaging\windows\pagex.iss"
            & $iscc (Join-Path $RepoRoot "packaging\windows\pagex.iss")
            if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
        } else {
            Write-Warning "Inno Setup 6 not found - installer not built (install: choco install innosetup)"
        }
    }

    # ---- report artifact sizes ------------------------------------------
    function Show-Size([string]$Path) {
        if (-not (Test-Path $Path)) { return }
        $item = Get-Item $Path
        $bytes = 0
        if ($item.PSIsContainer) {
            $sum = Get-ChildItem -Recurse -File -Path $Path |
                   Measure-Object -Property Length -Sum
            if ($sum.Sum) { $bytes = $sum.Sum }
        } else {
            $bytes = $item.Length
        }
        Write-Host ("  {0,-52} {1,10:N1} MB" -f $Path.Substring($RepoRoot.Length + 1), ($bytes / 1MB))
    }
    Write-Host ""
    Write-Host "artifacts:"
    Show-Size $gui
    Show-Size $cli
    Show-Size $setup
}
finally {
    Pop-Location
}
