# One command Docker (Windows) — requires Docker Desktop
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
Write-Host "Building + starting DubStudio (Dummy voice, ready to dub)..."
docker compose up --build -d
Write-Host ""
Write-Host "Open:   http://localhost:8080"
Write-Host "Health: http://localhost:8080/api/v1/health"
Write-Host "Docs:   http://localhost:8080/docs"
Write-Host "Logs:   docker compose logs -f"
Write-Host "Stop:   docker compose down"
