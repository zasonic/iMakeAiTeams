; installer-lite.iss — Inno Setup script for the LITE Windows installer.
; Bundles Tier 1 only (pywebview, anthropic, numpy, platformdirs, keyring,
; requests, psutil, tenacity, pydantic) — ~60 MB. Chat, agents, teams,
; router, security engine, prompt library all work. RAG, semantic search,
; and Telegram report unavailable via service_status() and the Settings
; → Subsystem status panel degrades them gracefully.
;
; Build pipeline:
;   set MYAI_VARIANT=lite
;   pyinstaller build\MyAIAgentHub.spec --noconfirm --clean
;   iscc build\installer-lite.iss
;
; Distinct AppId + AppName so the lite build coexists with the full build
; on the same machine (useful for switching or A/B testing). Users who want
; to migrate lite → full uninstall lite first; their data survives because
; paths.user_dir() resolves to the same %LOCALAPPDATA%\iMakeAiTeams\ for
; both builds.

#define AppName "MyAI Agent Hub Lite"
#define AppVersion "5.0.2"
#define AppPublisher "iMakeAiTeams"
#define AppURL "https://myaiagenthub.app"
#define AppExeName "MyAIAgentHub-lite.exe"

[Setup]
AppId={{B2C3D4E5-F6A7-8901-BCDE-F23456789012}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=dist
OutputBaseFilename=MyAIAgentHub-Setup-Lite
SetupIconFile=icons\AppIcon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "dist\MyAIAgentHub-lite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
