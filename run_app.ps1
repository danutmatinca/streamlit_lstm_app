# run_app.ps1 – startet die Streamlit-App (Windows PowerShell)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Kein venv gefunden. Starte setup_env.ps1 ..." -ForegroundColor Yellow
    & "$PSScriptRoot\setup_env.ps1"
}

. .\.venv\Scripts\Activate.ps1
Write-Host "Environment aktiv. Starte Streamlit ..." -ForegroundColor Green
streamlit run app.py
