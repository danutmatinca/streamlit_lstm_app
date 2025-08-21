# Third-Party Licenses / Drittanbieter-Lizenzen

Dieses Projekt nutzt Open-Source-Komponenten. Nachstehend eine Übersicht der wichtigsten Abhängigkeiten
aus `requirements.txt` und deren **upstream**-Lizenzen.  

Die vollständigen Lizenztexte liegen jeweils in den Projektrepositories der jeweiligen Pakete bzw. werden
zusammen mit den Python‑Paketen ausgeliefert (z. B. unter `site-packages/<pkg>/LICENSE*`).

> **Hinweis:** Diese Liste ist eine Orientierung und basiert auf den gängigen Lizenzen der Pakete.
> Für eine **versionsexakte, maschinell erzeugte** Liste kannst du innerhalb deiner virtuellen Umgebung
> `pip-licenses` verwenden:
> ```bash
> pip install pip-licenses
> pip-licenses --format=markdown --with-license-file --with-authors --ignore-packages pip setuptools wheel
> ```

---

## Paketübersicht

| Paket            | Version (mind.) | Lizenz         | Quelle |
|------------------|-----------------|----------------|--------|
| streamlit        | 1.36+           | Apache-2.0     | https://github.com/streamlit/streamlit |
| pandas           | 2.2.2           | BSD-3-Clause   | https://github.com/pandas-dev/pandas |
| numpy            | 1.26+           | BSD-3-Clause   | https://github.com/numpy/numpy |
| scikit-learn     | 1.4–1.5         | BSD-3-Clause   | https://github.com/scikit-learn/scikit-learn |
| tensorflow-cpu   | 2.16.1          | Apache-2.0     | https://github.com/tensorflow/tensorflow |
| yfinance         | 0.2.52+         | Apache-2.0     | https://github.com/ranaroussi/yfinance |
| matplotlib       | 3.8+            | Matplotlib License (BSD-kompatibel) | https://github.com/matplotlib/matplotlib |
| plotly           | 5.22+           | MIT            | https://github.com/plotly/plotly.py |
| python-dotenv    | 1.0+            | BSD-3-Clause   | https://github.com/theskumar/python-dotenv |
| protobuf         | <5.0            | BSD-3-Clause   | https://github.com/protocolbuffers/protobuf |
| typing-extensions| 4.5+            | PSF            | https://github.com/python/typing_extensions |

---

## Lizenz-Hinweise (Kurzformen)

- **Apache License 2.0** – Zulässige, patentfreundliche Lizenz. Erfordert u. a. die Beibehaltung von Copyright-
  und Lizenzhinweisen sowie ggf. NOTICE‑Hinweise.  
  Volltext: https://www.apache.org/licenses/LICENSE-2.0

- **BSD-3-Clause** – Zulässige Lizenz mit drei Klauseln. Erfordert die Beibehaltung von Copyright- und Lizenzhinweisen.  
  Info: https://opensource.org/license/bsd-3-clause/

- **MIT** – Sehr zulässige Lizenz. Erfordert die Beibehaltung des Copyright- und Lizenzhinweises.  
  Info: https://opensource.org/license/mit/

- **Matplotlib License (PSF-basiert, BSD‑kompatibel)** – Matplotlib steht unter einer PSF‑basierten, BSD‑kompatiblen Lizenz.  
  Info: https://matplotlib.org/stable/users/project/license.html

- **PSF License** – Lizenz der Python Software Foundation.  
  Info: https://docs.python.org/3/license.html

---

## Erfüllung von Lizenzpflichten

- Dieses Repository enthält die Projektlizenz (`LICENSE`).  
- Für Komponenten unter **Apache‑2.0** wird zusätzlich diese Datei (`THIRD_PARTY_LICENSES.md`) und eine `NOTICE` bereitgestellt.  
- Beachte, dass zusätzliche Lizenztexte einzelner Abhängigkeiten ggf. in deren Wheels/Source‑Distributions enthalten sind.
