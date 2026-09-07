$ErrorActionPreference = "Stop"

# Set working directory to project root
Set-Location $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "Error: Virtual environment not found at $pythonExe" -ForegroundColor Red
    Pause
    exit 1
}

# Determine available port (default: 8080, fallback: 8050 if 8080 is occupied by NVIDIA Broadcast etc.)
$port = 8080
$occupied = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue

if ($occupied) {
    $proc = Get-Process -Id $occupied[0].OwningProcess -ErrorAction SilentlyContinue
    $procDesc = if ($proc) { "$($proc.ProcessName) (PID $($proc.Id))" } else { "another application" }
    Write-Host "Note: Port 8080 is currently in use by $procDesc." -ForegroundColor Yellow
    Write-Host "Automatically switching to port 8050 for DubStudio..." -ForegroundColor Cyan
    $port = 8050
}

$url = "http://127.0.0.1:$port"

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "           DubStudio AI Video Dubbing Studio              " -ForegroundColor Green
Write-Host "           Running at: $url                               " -ForegroundColor White
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""

# Automatically open the browser
Start-Process $url

# Launch FastAPI / Uvicorn server directly with venv python
& $pythonExe -m uvicorn dubstudio.main:app --host 127.0.0.1 --port $port --reload