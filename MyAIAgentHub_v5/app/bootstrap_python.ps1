# bootstrap_python.ps1 — Download portable Python for Windows
param([string]$AppDir)

$PythonVersion = "3.11.9"
$Arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "win32" }
$Url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-$Arch.zip"
$PipUrl = "https://bootstrap.pypa.io/get-pip.py"

$TargetDir = Join-Path $AppDir ".python"
$ZipPath = Join-Path $env:TEMP "python-portable.zip"

Write-Host "Downloading Python $PythonVersion..."
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
} catch {
    Write-Error "Download failed: $_"
    exit 1
}

Write-Host "Extracting..."
if (Test-Path $TargetDir) { Remove-Item $TargetDir -Recurse -Force }
Expand-Archive -Path $ZipPath -DestinationPath $TargetDir -Force
Remove-Item $ZipPath -Force

# Enable pip in embedded Python by uncommenting import site in ._pth file
$PthFile = Get-ChildItem "$TargetDir\python*._pth" | Select-Object -First 1
if ($PthFile) {
    $content = Get-Content $PthFile.FullName
    $content = $content -replace "^#import site", "import site"
    Set-Content $PthFile.FullName $content
}

# Install pip
$GetPipPath = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri $PipUrl -OutFile $GetPipPath -UseBasicParsing
& "$TargetDir\python.exe" $GetPipPath --no-warn-script-location
Remove-Item $GetPipPath -Force

Write-Host "Python $PythonVersion installed to $TargetDir"
exit 0
