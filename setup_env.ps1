# setup_env.ps1 – erstellt venv und installiert Abhängigkeiten (Windows PowerShell)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Setup abgeschlossen. Starte: .\run_app.ps1" -ForegroundColor Green
