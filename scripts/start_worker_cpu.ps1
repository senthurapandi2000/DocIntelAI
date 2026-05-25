param(
    [int]$BatchSize = 1,
    [int]$NumBeams = 1,
    [float]$ConfidenceThreshold = 70,
    [int]$PollInterval = 5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .venv\Scripts\python.exe"
}

$env:CUDA_VISIBLE_DEVICES = "-1"

& ".venv\Scripts\python.exe" -m src.services.irs_processing_worker `
    --database data\processed\irs_review_queue.db `
    --poll-interval $PollInterval `
    --batch-size $BatchSize `
    --num-beams $NumBeams `
    --confidence-threshold $ConfidenceThreshold
