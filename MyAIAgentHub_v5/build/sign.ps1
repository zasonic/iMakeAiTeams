# sign.ps1 — Microsoft Trusted Signing wrapper.
#
# Usage:
#     pwsh build/sign.ps1 <path-to-artifact>
#
# Behavior:
#   - If MYAI_SIGN env var is unset or empty: no-op (exit 0). Keeps the
#     default build pipeline usable before the Azure Trusted Signing
#     subscription + service principal are provisioned.
#   - If MYAI_SIGN is set: invoke signtool.exe with the Trusted Signing
#     dlib. Fails hard if signtool is missing or the call errors.
#
# Required env when MYAI_SIGN is set:
#     TRUSTED_SIGNING_DLIB     Directory containing TrustedSigning.dll
#     TRUSTED_SIGNING_METADATA Path to signing-metadata.json
#
# Called twice per build (two-pass signing):
#     1. on dist\MyAIAgentHub\MyAIAgentHub.exe  (before Inno Setup packs it)
#     2. on dist\MyAIAgentHub-Setup-Full.exe    (after Inno Setup emits it)

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Target
)

$ErrorActionPreference = 'Stop'

if (-not $env:MYAI_SIGN) {
    Write-Host "[sign] MYAI_SIGN not set — skipping signing for $Target"
    exit 0
}

if (-not (Test-Path $Target)) {
    Write-Error "[sign] target not found: $Target"
    exit 1
}

$dlibDir = $env:TRUSTED_SIGNING_DLIB
$metadata = $env:TRUSTED_SIGNING_METADATA

if (-not $dlibDir -or -not (Test-Path (Join-Path $dlibDir 'TrustedSigning.dll'))) {
    Write-Error "[sign] TRUSTED_SIGNING_DLIB does not contain TrustedSigning.dll: $dlibDir"
    exit 1
}
if (-not $metadata -or -not (Test-Path $metadata)) {
    Write-Error "[sign] TRUSTED_SIGNING_METADATA missing: $metadata"
    exit 1
}

$signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
if (-not $signtool) {
    Write-Error "[sign] signtool.exe not on PATH. Install Windows 10/11 SDK."
    exit 1
}

$dlibPath = Join-Path $dlibDir 'TrustedSigning.dll'

Write-Host "[sign] signing $Target"
& $signtool.Source sign `
    /v /fd sha256 `
    /tr 'http://timestamp.acs.microsoft.com' /td sha256 `
    /dlib $dlibPath `
    /dmdf $metadata `
    $Target
if ($LASTEXITCODE -ne 0) {
    Write-Error "[sign] signtool failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[sign] verifying $Target"
& $signtool.Source verify /pa /v $Target
if ($LASTEXITCODE -ne 0) {
    Write-Error "[sign] verification failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[sign] ok: $Target"
