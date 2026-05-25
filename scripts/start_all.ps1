param(
    [ValidateSet("cpu", "gpu")]
    [string]$WorkerMode = "cpu"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ApiScript = Join-Path $PSScriptRoot "start_api.ps1"
$DashboardScript = Join-Path $PSScriptRoot "start_dashboard.ps1"

if ($WorkerMode -eq "gpu") {
    $WorkerScript = Join-Path $PSScriptRoot "start_worker_gpu.ps1"
}
else {
    $WorkerScript = Join-Path $PSScriptRoot "start_worker_cpu.ps1"
}

$RequiredScripts = @(
    $ApiScript,
    $WorkerScript,
    $DashboardScript
)

foreach ($ScriptPath in $RequiredScripts) {
    if (-not (Test-Path $ScriptPath)) {
        throw "Required startup script was not found: $ScriptPath"
    }
}

Start-Process `
    -FilePath "powershell.exe" `
    -WorkingDirectory $ProjectRoot `
    -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$ApiScript`""
    )

Start-Sleep -Seconds 2

Start-Process `
    -FilePath "powershell.exe" `
    -WorkingDirectory $ProjectRoot `
    -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$WorkerScript`""
    )

Start-Sleep -Seconds 2

Start-Process `
    -FilePath "powershell.exe" `
    -WorkingDirectory $ProjectRoot `
    -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$DashboardScript`""
    )

Write-Host ""
Write-Host "Started FastAPI, worker ($WorkerMode), and Streamlit."
Write-Host "API docs:  http://127.0.0.1:8000/docs"
Write-Host "Dashboard: http://localhost:8501"
