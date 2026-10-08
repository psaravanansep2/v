# Installs v, the free voice assistant, on Windows.
#
#   irm https://raw.githubusercontent.com/psaravanansep2/v/main/install.ps1 | iex
#
# What it does, all inside your user folder (no admin rights):
#   1. installs uv, a small tool that brings its own Python
#   2. installs v with it
#   3. downloads the free AI model that fits this computer (one time, a few GB)
#   4. adds v to the Start menu and Desktop, and opens it
#
# Options (environment variables): V_KOKORO=1 adds the most natural voice
# (a larger download); V_SKIP_MODEL=1 skips step 3 (v does it on first launch).
$ErrorActionPreference = 'Stop'

$Repo = if ($env:V_REPO) { $env:V_REPO } else { 'https://github.com/psaravanansep2/v' }
$Ref = if ($env:V_REF) { $env:V_REF } else { 'main' }
$Source = if ($env:V_SOURCE) { $env:V_SOURCE } else { "$Repo/archive/refs/heads/$Ref.zip" }
$Extras = if ($env:V_EXTRAS) { $env:V_EXTRAS } else { 'all' }
$PythonVersion = if ($env:V_PYTHON) { $env:V_PYTHON } else { '3.12' }
if ($env:V_KOKORO -eq '1') { $Extras = "$Extras,kokoro" }

function Step($text) { Write-Host ""; Write-Host $text -ForegroundColor Cyan }

$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Step '1/4  Getting uv (brings its own Python)...'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    try {
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    } catch {
        Write-Host "uv's installer failed: $_"
    }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "Couldn't install uv. Check the internet connection and try again."
    }
}

Step '2/4  Installing v...'
uv tool install --force --python $PythonVersion "v[$Extras] @ $Source"
if ($LASTEXITCODE -ne 0) { throw 'Installing v failed (see above). Check the internet connection and try again.' }
# Make sure new terminals find v. uv prints notes on stderr, which Windows
# PowerShell would otherwise treat as an error, so let them pass quietly.
$ErrorActionPreference = 'Continue'
& uv tool update-shell *> $null
$ErrorActionPreference = 'Stop'
$env:Path = "$(& uv tool dir --bin);$env:Path"

if ($env:V_SKIP_MODEL -ne '1') {
    Step '3/4  Getting the free AI model ready (one time; this can take a while)...'
    v setup
    if ($LASTEXITCODE -ne 0) { Write-Host "The model download didn't finish; v will try again when it starts." }
}

Step '4/4  Adding v to the Start menu and Desktop...'
v install-shortcut

Write-Host ''
Write-Host 'Done! v is ready.' -ForegroundColor Green
Write-Host 'Open it any time from the Start menu or Desktop.'
Write-Host "Use it from your phone: click the phone button in v's window and scan the code."
if ($env:V_NO_LAUNCH -ne '1') {
    $link = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\v.lnk'
    if (Test-Path $link) { Start-Process $link }
}
