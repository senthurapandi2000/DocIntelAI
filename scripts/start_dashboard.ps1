param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .venv\Scripts\python.exe"
}

$env:DOCINTELAI_REVIEW_API = "http://127.0.0.1:8000"

& ".venv\Scripts\python.exe" -m streamlit run `
    src\ui\irs_review_dashboard.py `
    --server.port $Port
