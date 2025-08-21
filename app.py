# SPDX-FileCopyrightText: © 2025 Danut Matinca
# SPDX-License-Identifier: MIT

import os
import datetime as dt
from typing import Sequence, Tuple, Any, cast

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -- Page setup & Theme tweaks --

st.set_page_config(
    page_title="KI-Börsenprognose (LSTM) – Demo",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"  # wichtig: Toggle bleibt sichtbar & startet ausgeklappt
)

# --- Sidebar-Buttons: gleiche Breite + kein Zeilenumbruch ---
st.markdown("""
<style>
  [data-testid="stSidebar"] { width: 16rem; min-width: 16rem; }  /* 16–18rem testen */
  [data-testid="stSidebar"] .stButton > button {
    width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '''
    <style>
    .stApp {background: linear-gradient(180deg, #0f172a 0%, #111827 100%)}
    .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stDateInput, .stButton, .stCheckbox, .stSlider, .stFileUploader { color: #e5e7eb !important; }

    /* WICHTIG: Header NICHT verstecken, sonst verschwindet der << / >>-Toggle!
       Stattdessen nur Menü & Footer ausblenden: */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}

    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .metric {background: #111827; border-radius: 1rem; padding: .75rem 1rem; border: 1px solid #1f2937;}
    .card {background: #0b1220; border: 1px solid #1f2937; border-radius: 1rem; padding: 1rem;}
    </style>
    ''', unsafe_allow_html=True
)

st.write("")  # eine leere Zeile
st.markdown(
    '''
    <p style="font-size:20px;">Dieses Projekt dient ausschließlich zu Lern-, Forschungs- und Demonstrationszwecken.</p>
    <br><br>
    ''',
    unsafe_allow_html=True
)

# -- Sidebar --
st.sidebar.title("⚙️ Einstellungen")
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="AAPL")

start_date = st.sidebar.date_input("Startdatum", value=dt.date.today() - dt.timedelta(days=365 * 5))
end_date = st.sidebar.date_input("Enddatum", value=dt.date.today())

# erzwinge ints (Slider liefern zwar ints, Casting verhindert Typ-Vererbung/Inspections)
lookback_days = cast(int, st.sidebar.slider("Lookback (Tage)", min_value=10, max_value=120, value=60, step=5))
test_size_percent = cast(int, st.sidebar.slider("Testanteil (%)", min_value=5, max_value=40, value=20, step=5))
num_epochs = cast(int, st.sidebar.slider("Epochen", min_value=1, max_value=25, value=5))
batch_sz = cast(int, st.sidebar.select_slider("Batchgröße", options=[16, 32, 64, 128], value=32))
do_train = st.sidebar.button(" 🏃‍♂️‍➡️ Trainieren & 🔮 Prognose")  # 🔖 Unicode-Emojis, per copy-paste, https://emojipedia.org/

st.title("📈 📉 KI-Börsenprognose – LSTM Demo")
st.caption("Interaktive Streamlit-App: Daten laden → LSTM trainieren → Vorhersage visualisieren")


# -- Data load --
@st.cache_data(show_spinner=False)
def load_data(tck: str, start_dt: dt.date, end_dt: dt.date) -> pd.DataFrame:
    df = yf.download(tck, start=start_dt, end=end_dt, auto_adjust=True)
    df = df.dropna()
    return df


with st.spinner("Lade Kursdaten ..."):
    data_df = load_data(ticker, start_date, end_date)

if data_df.empty:
    st.error("Keine Daten gefunden. Bitte Ticker oder Zeitraum prüfen.")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.subheader(f"{ticker} - Kursverlauf (Historie)")
    st.line_chart(data_df["Close"], height=280)
with right:
    st.subheader("Dateninfo")
    last_close_val = data_df["Close"].iloc[-1].item()
    st.markdown(f"<div class='metric'><b>Letzter Schlusskurs:</b><br>{last_close_val:,.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><b>Zeitraum:</b><br>{data_df.index.min().date()} → {data_df.index.max().date()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><b>Anzahl Datenpunkte:</b><br>{len(data_df):,}</div>", unsafe_allow_html=True)


# -- Sequence builder --
def build_sequences(values: Sequence[float] | np.ndarray, lookback_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Erzeuge überlappende Sequenzen (X) und Ziele (y) aus 1D-Werten."""
    lb = int(lookback_window)  # Robust gegen str aus Widgets
    vals = np.asarray(values, dtype=float).flatten()

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for i in range(lb, len(vals)):
        X_list.append(vals[i - lb:i])
        y_list.append(vals[i])

    X_arr = np.asarray(X_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)
    return X_arr[..., np.newaxis], y_arr  # Feature-Dimension hinzufügen


# -- Train + Predict --
def train_and_predict(df: pd.DataFrame, lookback_window: int = 60, test_ratio: float = 0.2,
                      epochs_num: int = 5, batch_size_val: int = 32) -> dict[str, Any]:
    # Nur "Close" verwenden
    close = df["Close"].to_numpy().reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close).flatten()

    X, y = build_sequences(scaled, lookback_window)
    split_idx = int(len(X) * (1 - float(test_ratio)))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    model = keras.Sequential([
        layers.Input(shape=(lookback_window, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=int(epochs_num),
        batch_size=int(batch_size_val),
        verbose=0,
        callbacks=callbacks
    )

    # Vorhersagen
    preds_test = model.predict(X_test, verbose=0).flatten()

    # Skalierung zurückdrehen
    inv_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    inv_preds = scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()

    # T+1 Forecast aus letztem Fenster
    last_window = scaled[-int(lookback_window):]
    next_pred_scaled = model.predict(last_window.reshape(1, int(lookback_window), 1), verbose=0).flatten()[0]
    next_pred = float(scaler.inverse_transform([[next_pred_scaled]])[0, 0])

    return {
        "model": model,
        "history": history.history,
        "y_true": inv_test,
        "y_pred": inv_preds,
        "next_pred": next_pred,
        "split_index": split_idx
    }


# -- UI: Train button --
if do_train:
    with st.spinner("Trainiere LSTM und berechne Vorhersage ..."):
        res = train_and_predict(
            data_df,
            lookback_window=lookback_days,
            test_ratio=test_size_percent / 100.0,
            epochs_num=num_epochs,
            batch_size_val=batch_sz
        )

    st.subheader("Vorhersage (Testset)")
    # Index des Testbereichs ausrichten
    test_idx = data_df.index[lookback_days + res["split_index"]: lookback_days + res["split_index"] + len(res["y_true"])]
    pred_df = pd.DataFrame({"Wahr": res["y_true"], "Prognose": res["y_pred"]}, index=test_idx)
    st.line_chart(pred_df, height=320)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Nächster prognostizierter Schlusskurs (T+1):**")
        st.success(f"{res['next_pred']:,.2f}")
    with col2:
        last_close_val: float = float(data_df["Close"].iloc[-1])
        delta = res['next_pred'] - last_close_val
        direction = "▲" if delta >= 0 else "▼"
        st.write("**Delta zur letzten Close:**")
        st.info(f"{direction} {delta:,.2f}")

    with st.expander("Trainingsdetails"):
        # numpy/pandas-runden statt built-in round
        loss_list = [float(x) for x in np.round(res["history"].get("loss", []), 6)]
        val_loss_list = [float(x) for x in np.round(res["history"].get("val_loss", []), 6)]
        st.json({
            "loss": loss_list,
            "val_loss": val_loss_list,
            "lookback": lookback_days,
            "test_ratio": test_size_percent / 100.0
        })

st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color: #9ca3af; line-height: 1.4; margin-top: 2rem;'>
        <p>
            Dieses Projekt dient ausschließlich Studien-, Forschungs- und Demonstrationszwecken.<br>
            Es handelt sich nicht um ein Finanzwerkzeug und bietet keinerlei finanzielle, steuerliche oder rechtliche Beratung.<br>
            Jegliche Nutzung erfolgt auf eigene Verantwortung. Der Autor übernimmt keine Haftung für Schäden oder Verluste,
            die aus der Anwendung des Programms entstehen.
        </p>
        <br>
        <p>
            © 2025 Danut Matinca – Powered by Streamlit · Third-Party Licenses siehe THIRD_PARTY_LICENSES.md
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


