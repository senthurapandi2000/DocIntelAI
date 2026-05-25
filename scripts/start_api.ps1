param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .venv\Scripts\python.exe"
}

& ".venv\Scripts\python.exe" -m uvicorn `
    src.api.irs_review_api:app `
    --host $HostAddress `
    --port $Port `
    --reload
