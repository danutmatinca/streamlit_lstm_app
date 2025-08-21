#!/bin/bash
# setup_env.sh – erstellt venv und installiert Abhängigkeiten
set -e
cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 nicht gefunden. Bitte installieren (z. B. via deadsnakes PPA unter Ubuntu)."
  exit 1
fi

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Setup abgeschlossen. Starte: ./run_app.sh"
