#!/bin/bash
# run_app.sh – startet die Streamlit-App
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Kein .venv gefunden. Starte setup_env.sh ..."
  bash setup_env.sh
fi

source .venv/bin/activate
echo "Environment aktiv. Starte Streamlit ..."
exec streamlit run app.py
