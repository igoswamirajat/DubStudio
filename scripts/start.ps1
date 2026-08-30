# Local start without Docker
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
pip install -q -e ".[dev]"
$env:DUBSTUDIO_TTS_ENGINE = if ($env:DUBSTUDIO_TTS_ENGINE) { $env:DUBSTUDIO_TTS_ENGINE } else { "dummy" }
Write-Host "API http://127.0.0.1:8080  Docs /docs"
uvicorn dubstudio.main:app --host 127.0.0.1 --port 8080 --reload
