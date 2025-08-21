# KI-Börsenprognose – LSTM Demo (Streamlit)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge.svg)](https://lstm-ai-predictor.streamlit.app)

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/danut-matinca/streamlit_lstm_app?style=social)

Interaktive App: Kursdaten von Yahoo Finance laden → LSTM (TensorFlow) trainieren → Vorhersage und Metriken anzeigen.  
Nur zu Lern-, Forschungs- und Demonstrationszwecken – keine Anlageberatung.  

![App Screenshot](screenshot_app.png)  

Autor: Danut Matinca · E-Mail: <danut.matinca@yahoo.com>  

---

## Was macht dieses Programm?
Diese App lädt historische Schlusskurse eines Tickers (z. B. AAPL) mit `yfinance`, skaliert die Werte und trainiert ein LSTM-Netz zur Ein-Schritt-Vorhersage (T+1).  
In der Sidebar stellst du Ticker, Zeitraum, Lookback, Testanteil, Epochen und Batchgröße ein und startest per Button „Trainieren & Prognose“.  

### Funktionen
- Daten laden (yfinance, `auto_adjust=True`) und bereinigen  
- Skalierung mit `MinMaxScaler(0..1)`  
- Sequenzen aus Lookback-Fenstern erstellen  
- LSTM-Modell: 64-Units → Dropout → 32-Units → Dense(1), `loss='mse'`, `optimizer='adam'`  
- EarlyStopping auf `val_loss` (patience=3)  
- Vorhersagen für Testset + nächster Schlusskurs (T+1)  
- Visualisierung: „Wahr“ vs. „Prognose“; Delta zum letzten Close  
- Trainingsdetails (Loss/ValLoss) im Expander  
- Fehlerbehandlung bei leeren Daten  

### Kurz zu LSTM
Long Short‑Term Memory (LSTM) ist eine rekurrente Netzarchitektur, die mittels Gates (Input/Forget/Output) Langzeitabhängigkeiten in Sequenzen lernen kann.  
Die Demo nutzt ein Sliding‑Window der letzten N Schlusskurse (Lookback) als Eingabe und schätzt daraus den nächsten Close‑Preis (Single‑Feature‑Forecast).


---

## Installation und Start

### Linux / macOS
```bash
# 1) Virtuelles Environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2) Abhängigkeiten
pip install --upgrade pip
pip install -r requirements.txt

# 3) Start
streamlit run app.py
```

### Windows (PowerShell)
```powershell
# 1) Virtuelles Environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Abhängigkeiten
pip install --upgrade pip
pip install -r requirements.txt

# 3) Start
streamlit run app.py
```

**Tipp:** Alternativ die Skripte `setup_env.(sh|ps1)` und `run_app.(sh|ps1)` verwenden.  

---

## Projektstruktur
```text
streamlit_lstm_app/
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── TREE.md
├── run_app.sh
├── setup_env.sh
├── run_app.ps1
├── setup_env.ps1
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Skripte (Erklärung)
- **setup_env.sh** (Linux/macOS): Erzeugt `.venv` (Python 3.11), aktiviert es und installiert alle Abhängigkeiten aus `requirements.txt`.  
- **run_app.sh** (Linux/macOS): Aktiviert `.venv` (führt bei Bedarf vorher `setup_env.sh` aus) und startet die App mit `streamlit run app.py`.  
- **setup_env.ps1** (Windows/PowerShell): Entspricht `setup_env.sh` für Windows. Legt `.venv` mit Python 3.11 an und installiert Requirements.  
- **run_app.ps1** (Windows/PowerShell): Entspricht `run_app.sh` für Windows. Aktiviert `.venv` und startet die App.  

Tipp: Nach dem Setup reicht künftig `./run_app.sh` bzw. `.
un_app.ps1`.  

---

## Warum ist der Ordner lokal groß?
Hauptsächlich wegen des virtuellen Environments (`.venv/`) und großer Binärpakete (z. B. TensorFlow).  
Das wird nicht eingecheckt – `.gitignore` schließt `.venv/` aus. Auf GitHub bleibt das Repo klein.  

---

## Sprachen-Anteile im Repo
Auf GitHub automatisch via „GitHub Linguist“ (Farbbalken oben).  
Lokal messen:  
```bash
sudo apt install -y cloc
cloc .
```

---

## Third-Party Licenses
Dieses Projekt verwendet Open-Source-Bibliotheken (u. a. Streamlit, pandas, NumPy, scikit-learn, TensorFlow, yfinance, Matplotlib, Plotly).  
Eine Übersicht der Lizenzen findest du unter **[THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md)**.  
Zusätzliche Hinweise für Apache-2.0-Komponenten stehen in **[NOTICE](./NOTICE)**.  

> Für eine **versionsexakte** Liste nach der Installation aller Abhängigkeiten:  
> ```bash
> pip install pip-licenses
> pip-licenses --format=markdown --with-license-file --with-authors --ignore-packages pip setuptools wheel
> ```

---

## Hinweise zum UI
- UI & App-Code © 2025 Danut Matinca.  
- „Powered by Streamlit“: das Framework ist Apache-2.0; siehe Third-Party-Lizenzen.  

---

## Lizenz
Dieses Repository steht unter der **MIT-Lizenz** – siehe **[LICENSE](./LICENSE)**.  
© 2025 Danut Matinca. Drittanbieter-Lizenzen siehe **[THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md)**.  
