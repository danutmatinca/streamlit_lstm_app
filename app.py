import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -- Page setup & Theme tweaks --
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.set_page_config(initial_sidebar_state="expanded")
st.set_page_config(page_title="KI-Börsenprognose (LSTM) – Demo", layout="wide", page_icon="📈")
st.markdown(
    '''
    <style>
    .stApp {background: linear-gradient(180deg, #0f172a 0%, #111827 100%)}
    .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stDateInput, .stButton, .stCheckbox, .stSlider, .stFileUploader { color: #e5e7eb !important; }
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .metric {background: #111827; border-radius: 1rem; padding: .75rem 1rem; border: 1px solid #1f2937;}
    .card {background: #0b1220; border: 1px solid #1f2937; border-radius: 1rem; padding: 1rem;}
    </style>
    ''', unsafe_allow_html=True
)

st.markdown(
    """
    <p style="font-size:18px;">Dieses Projekt dient ausschließlich zu Lern-, Forschungs- und Demonstrationszwecken.</p>
    """,
    unsafe_allow_html=True
)

# -- Sidebar --
st.sidebar.title("⚙️ Einstellungen")
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="AAPL")
start = st.sidebar.date_input("Startdatum", value=dt.date.today() - dt.timedelta(days=365*5))
end = st.sidebar.date_input("Enddatum", value=dt.date.today())
lookback = st.sidebar.slider("Lookback (Tage)", min_value=10, max_value=120, value=60, step=5)
test_size = st.sidebar.slider("Testanteil (%)", min_value=5, max_value=40, value=20, step=5)
epochs = st.sidebar.slider("Epochen", min_value=1, max_value=25, value=5)
batch_size = st.sidebar.select_slider("Batchgröße", options=[16, 32, 64, 128], value=32)
do_train = st.sidebar.button(" 🏋️ Trainieren & 🔭 Vorhersagen")        # 🔖 Unicode-Emojis, per copy-paste,  https://emojipedia.org/

st.title("📈 📉 KI-Börsenprognose – LSTM Demo")
st.caption("Interaktive Streamlit-App: Daten laden → LSTM trainieren → Vorhersage visualisieren")

# -- Data load --
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    # df = yf.download(ticker, start=start, end=end)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df.dropna()
    return df

with st.spinner("Lade Kursdaten ..."):
    data = load_data(ticker, start, end)

if data.empty:
    st.error("Keine Daten gefunden. Bitte Ticker oder Zeitraum prüfen.")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Historische Kurse – {ticker}")
    st.line_chart(data["Close"], height=280)
with right:
    st.subheader("Dateninfo")
    # last_close = float(data["Close"].iloc[-1])
    last_close = data["Close"].iloc[-1].item()
    st.markdown(f"<div class='metric'><b>Letzter Schlusskurs:</b><br>{last_close:,.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><b>Zeitraum:</b><br>{data.index.min().date()} → {data.index.max().date()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><b>Anzahl Datenpunkte:</b><br>{len(data):,}</div>", unsafe_allow_html=True)

# -- Sequence builder --
def build_sequences(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y  # add feature dim

# -- Train + Predict --
def train_and_predict(df, lookback=60, test_ratio=0.2, epochs=5, batch_size=32):
    # Use only Close for a simple demo
    close = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close).flatten()

    X, y = build_sequences(scaled, lookback)
    split = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
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
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    # Predictions
    preds_test = model.predict(X_test, verbose=0).flatten()
    # invert scaling
    inv_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    inv_preds = scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()

    # next-day forecast using last lookback window
    last_window = scaled[-lookback:]
    next_pred_scaled = model.predict(last_window.reshape(1, lookback, 1), verbose=0).flatten()[0]
    next_pred = float(scaler.inverse_transform([[next_pred_scaled]])[0, 0])

    return {
        "model": model,
        "history": history.history,
        "y_true": inv_test,
        "y_pred": inv_preds,
        "next_pred": next_pred,
        "split_index": split
    }

# -- UI: Train button --
if do_train:
    with st.spinner("Trainiere LSTM und berechne Vorhersage ..."):
        res = train_and_predict(data, lookback=lookback, test_ratio=test_size/100, epochs=epochs, batch_size=batch_size)

    st.subheader("Vorhersage (Testset)")
    # Align the test index
    test_index = data.index[lookback + res["split_index"] : lookback + res["split_index"] + len(res["y_true"])]
    pred_df = pd.DataFrame({
        "Wahr": res["y_true"],
        "Prognose": res["y_pred"]
    }, index=test_index)

    st.line_chart(pred_df, height=320)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Nächster prognostizierter Schlusskurs (T+1):**")
        st.success(f"{res['next_pred']:,.2f}")
    with col2:
        last_close = float(data['Close'].iloc[-1])
        delta = res['next_pred'] - last_close
        direction = "▲" if delta >= 0 else "▼"
        st.write("**Delta zur letzten Close:**")
        st.info(f"{direction} {delta:,.2f}")

    with st.expander("Trainingsdetails"):
        st.json({
            "loss": [round(x, 6) for x in res["history"].get("loss", [])],
            "val_loss": [round(x, 6) for x in res["history"].get("val_loss", [])],
            "lookback": lookback,
            "test_ratio": test_size/100
        })

st.caption("Demo-Zwecke: Kein Finanz- oder Anlageberatung. Modelle sind vereinfacht und nur zur Veranschaulichung.")
