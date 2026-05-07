; installer.iss — Inno Setup script for the FULL Windows installer.
; Bundles Tier 1 + Tier 2 (sentence-transformers, torch-cpu, chromadb,
; rank-bm25) PLUS the all-MiniLM-L6-v2 embedding model (~90 MB) so RAG
; and semantic search work offline on first launch. Also bundles the
; Microsoft Edge WebView2 Runtime offline installer (~130 MB) and runs
; it conditionally so pre-22H2 Windows 11 clean VMs work out of the box.
;
; Build pipeline (Windows build host):
;   python build\fetch_model.py
;   set MYAI_VARIANT=full
;   pyinstaller build\MyAIAgentHub.spec --noconfirm --clean
;   iscc build\installer.iss
;
; Signing (optional, gated on MYAI_SIGN in build_windows.bat):
;   pwsh build\sign.ps1 dist\MyAIAgentHub\MyAIAgentHub.exe
;   iscc build\installer.iss
;   pwsh build\sign.ps1 dist\MyAIAgentHub-Setup-Full.exe

#define AppName "MyAI Agent Hub"
#define AppVersion "5.0.3"
#define AppPublisher "iMakeAiTeams"
#define AppURL "https://myaiagenthub.app"
#define AppExeName "MyAIAgentHub.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=MyAIAgentHub-Setup-Full
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
Source: "..\dist\MyAIAgentHub\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
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
  // Evergreen WebView2 Runtime product key. If pv is present and non-empty
  // the runtime is already installed — skip the bundled installer.
  Result := True;
  if RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
    if (Version <> '') and (Version <> '0.0.0.0') then
      Result := False;
  if Result then
    if RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) then
      if (Version <> '') and (Version <> '0.0.0.0') then
        Result := False;
end;
