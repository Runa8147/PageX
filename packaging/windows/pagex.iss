; ===========================================================================
; PageX Windows installer (Inno Setup 6)  ->  dist\PageX-Setup-x64.exe
;
; HOW TO BUILD (from the repository root, on Windows):
;
;   1. powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
;        - builds dist\pagex-gui\  (from pagex-gui.spec) and
;                  dist\pagex\     (from pagex-cli.spec)
;        - compiles this .iss automatically when ISCC.exe is installed
;   2. manual compile (equivalent):
;
;        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\pagex.iss
;
; SCOPE: the installer ships the frozen GUI onedir + config + samples only.
; Model weights (GGUF, ~5 GB) are deliberately NOT installed here - they are
; downloaded on first run by the in-app model wizard (`pagex models
; download`, resumable and SHA-256 verified) or installed offline with
; `pagex models install-pack <zip>`. That keeps the setup file in the low
; tens of MB. See docs/packaging.md.
;
; Per-user install (PrivilegesRequired=lowest): program files go to
; {localappdata}\PageX, while all *state* (SQLite db, artifacts, models,
; logs) lives under %LOCALAPPDATA%\pagex\ (platformdirs) and therefore
; survives uninstalling this app.
; ===========================================================================
#define AppName "PageX"
#define AppVersion "1.0.0"
#define AppExe "pagex-gui.exe"

[Setup]
AppId={{9F4C1D62-7E5B-4A38-BD90-6C2F81A5E417}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=PageX Team
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=PageX-Setup-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
LicenseFile=..\..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
  GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; the whole onedir build (pagex-gui.exe + _internal\)
Source: "..\..\dist\pagex-gui\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\PageX"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall PageX"; Filename: "{uninstallexe}"
Name: "{autodesktop}\PageX"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch PageX"; \
  Flags: nowait postinstall skipifsilent

; NOTE: the uninstaller removes {app} only. User data (db, artifacts, models
; and pagex.toml under %LOCALAPPDATA%\pagex) is intentionally kept - delete
; that folder manually for a full reset.
