$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .venv\Scripts\python.exe"
}

& ".venv\Scripts\python.exe" -m pytest tests -v
