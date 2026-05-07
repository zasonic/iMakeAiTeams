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
; on the same machine. User data at %LOCALAPPDATA%\MyAIAgentHub\ is shared
; because paths.user_dir() resolves to the same dir for both builds.

#define AppName "MyAI Agent Hub Lite"
#define AppVersion "5.0.3"
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
OutputDir=..\dist
OutputBaseFilename=MyAIAgentHub-Setup-Lite
SetupIconFile=..\icons\AppIcon.ico
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
Source: "..\dist\MyAIAgentHub-lite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "webview2\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; Parameters: "/silent /install"; StatusMsg: "Installing Microsoft Edge WebView2 Runtime..."; Check: NeedsWebView2; Flags: waituntilterminated
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{localappdata}\MyAIAgentHub\launch.log"

[Code]
function NeedsWebView2(): Boolean;
var
  Version: String;
begin
  Result := True;
  if RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
    if (Version <> '') and (Version <> '0.0.0.0') then
      Result := False;
  if Result then
    if RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
      if (Version <> '') and (Version <> '0.0.0.0') then
        Result := False;
end;
