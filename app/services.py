"""
services.py — Funciones de backend para InverBot
=================================================
Este módulo centraliza utilidades para:

* Normalizar timestamps a la zona horaria de Nueva York.
* Cargar modelos y escaladores entrenados por ticker.
* Generar y evaluar predicciones históricas (2 años) y de último mes.
* Descargar datos OHLCV con `yfinance` y sentimientos FinBERT.
* Exponer DataFrames listos para su consumo por la API Flask.

Todas las funciones están documentadas en español para mejorar la
mantenibilidad del proyecto.
"""

import pandas as pd
from datetime import date, timedelta, datetime
import os, joblib
import numpy as np
import requests
from bs4 import BeautifulSoup, FeatureNotFound
import pytz
from transformers import pipeline as hf_pipeline

# ------------------------------------------------------------------
#  Helper: Normalize timestamps to New York time (UTC‑05/‑04)
# ------------------------------------------------------------------
def _to_et_series(series: pd.Series) -> pd.Series:
    """
    Convierte una serie de fechas/horas al huso horario
    `America/New_York` y devuelve objetos `datetime` “naive”
    (sin información de zona) listos para graficarse en el front‑end.
    """
    if series.empty:
        return series
    # Parse if strings
    if series.dtype == object:
        series = pd.to_datetime(series, errors='coerce')
    # Localize naive ts to UTC, then convert
    if series.dt.tz is None or str(series.dt.tz) == 'None':
        series = series.dt.tz_localize('UTC')
    return series.dt.tz_convert('America/New_York').dt.tz_localize(None)


# Devuelve lista de tickers disponibles
BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE,'modelos')

def list_models():
    """Devuelve una lista con los tickers (carpetas) disponibles en `modelos/`."""
    return [d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS,d))]

# Añade docstring para get_prediction_df
def get_prediction_df(ticker, ds):
    """
    Carga desde disco el CSV con precios + predicciones históricas
    (`dbFinal_2y.csv` o `db_NOSeen_1mo.csv`) y lo transforma al formato
    esperado por el front‑end: columnas `date`, `real`, `pred`, además
    de adjuntar métricas de desempeño y la próxima predicción horaria.
    """
    file = "dbFinal_2y.csv" if ds == "2y" else "db_NOSeen_1mo.csv"
    csv_path = os.path.join(MODELS, ticker, file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df_raw = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # If the CSV does NOT contain any prediction column,
    # create predictions first *before* renombrar columnas
    # ------------------------------------------------------------
    has_pred = any(col.lower().startswith("pred") for col in df_raw.columns)
    if not has_pred:
        df_final = _add_predictions_to_df(df_raw.copy(), ticker)
    else:
        # Si ya traía predicciones, sólo normalizamos nombres
        rename_map = {
            "ts_hour": "date",
            "Date": "date",
            "Close": "real",
            "Real": "real",
            "Pred_Close": "pred",
            "Pred": "pred",
        }
        df_final = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})
        # Ajustar a hora de Nueva York
        df_final["date"] = _to_et_series(df_final["date"])
        df_final = df_final[["date", "real", "pred"]]

    # --- Add metrics & next prediction
    df_final = _add_metrics_and_next(df_final, ticker)
    return df_final
# --- Metrics & next prediction helper ---
def _add_metrics_and_next(df: pd.DataFrame, ticker: str):
    """
    Calcula MAE, MSE, RMSE, R² y MAPE entre `real` y `pred`.
    También intenta inferir la siguiente predicción usando el modelo
    almacenado; si falla, se toma el último valor de `pred`.
    Los resultados se guardan en `df.attrs`.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # Only use rows where both real and pred are not null
    mask = (~pd.isnull(df["real"])) & (~pd.isnull(df["pred"]))
    real = df.loc[mask, "real"].values
    pred = df.loc[mask, "pred"].values
    metrics = {}
    if len(real) > 0 and len(pred) > 0:
        metrics["mae"] = float(np.round(mean_absolute_error(real, pred), 4))
        metrics["mse"] = float(np.round(mean_squared_error(real, pred), 4))
        metrics["rmse"] = float(np.round(np.sqrt(mean_squared_error(real, pred)), 4))
        metrics["r2"] = float(np.round(r2_score(real, pred), 4))
        metrics["mape"] = float(np.round(np.mean(np.abs((real - pred) / real)) * 100, 4))
    else:
        metrics = {"mae": None, "mse": None, "rmse": None, "r2": None}

    # Next prediction: try to load model and predict next value if possible
    try:
        next_pred = _predict_next(df, ticker)
    except Exception:
        next_pred = None
    # Fallback: use último valor de la columna pred si el modelo no generó next_pred
    if next_pred is None and len(df["pred"]) > 0:
        next_pred = float(df["pred"].iloc[-1])

    # Attach as attributes for API to use (or as columns if needed)
    df.attrs["metrics"] = metrics
    df.attrs["next_pred"] = next_pred
    return df

# --- Helper to predict next value using last SEQ_LEN rows
def _predict_next(df: pd.DataFrame, ticker: str):
    """
    Usa las últimas `SEQ_LEN` ventanas de características para
    predecir el próximo precio de cierre.  Requiere que el modelo,
    escaladores y CSV completo de features existan en disco.
    Devuelve `None` si falta algún artefacto.
    """
    # Try to load the original csv with features if possible
    # Fallback: use last known pred + diff
    model_path = os.path.join(MODELS, ticker, "modelo.keras")
    scaler_path = os.path.join(MODELS, ticker, "scaler.save")
    close_scaler_path = os.path.join(MODELS, ticker, "close_scaler.save")
    for p in (model_path, scaler_path, close_scaler_path):
        if not os.path.exists(p):
            return None
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    close_scaler = joblib.load(close_scaler_path)
    # Try to load the original csv with all features
    # (assume it's the same as used for this prediction)
    # First, try dbFinal_2y.csv or db_NOSeen_1mo.csv
    # Use the same file as in get_prediction_df
    # This is a bit hacky, but works for our use case
    # Get the last SEQ_LEN rows of features
    # Try to find a csv with all feature cols
    for fname in ["dbFinal_2y.csv", "db_NOSeen_1mo.csv", "db_last_month.csv"]:
        fpath = os.path.join(MODELS, ticker, fname)
        if os.path.exists(fpath):
            df_full = pd.read_csv(fpath)
            break
    else:
        return None
    # Ensure all needed columns
    for col in FEATURE_COLS:
        if col not in df_full.columns:
            df_full[col] = 0.0
    df_full["Stock_Split_Flag"] = (
        df_full.get("Stock_Split_Flag")
        if "Stock_Split_Flag" in df_full.columns
        else (df_full.get("Stock Splits", 0) != 0).astype(int)
    )
    df_full["Volume"] = np.log1p(df_full["Volume"])
    # Get last SEQ_LEN rows
    if len(df_full) < SEQ_LEN:
        return None
    X_last = df_full[FEATURE_COLS].values[-SEQ_LEN:]
    X_last_scaled = scaler.transform(X_last[:-1])  # all but last
    # To predict the next, need a single window of SEQ_LEN rows
    X_seq = np.vstack([X_last_scaled, scaler.transform(X_last[-1:])])
    # But shape must be (1, SEQ_LEN, features)
    X_seq = np.expand_dims(X_last, axis=0)
    X_seq_scaled = scaler.transform(X_last)
    X_seq_scaled = np.expand_dims(X_seq_scaled, axis=0)
    pred_scaled = model.predict(X_seq_scaled, verbose=0).reshape(-1, 1)
    pred_real = close_scaler.inverse_transform(pred_scaled).flatten()
    return float(np.round(pred_real[0], 4))

# ---------------------------------------------------------------------------
#  FULL prediction pipeline for an **uploaded** CSV
# ---------------------------------------------------------------------------
import tensorflow as tf

SEQ_LEN = 7
FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "cnt_articles",
    "sum_weighted",
    "Stock_Split_Flag",
]

def _create_sequences(df, seq_len=SEQ_LEN):
    """
    Constrye ventanas deslizantes de longitud `seq_len` empleadas
    durante el entrenamiento LSTM. Devuelve `np.ndarray` con forma
    `(n_samples, seq_len, n_features)`.
    """
    data = df[FEATURE_COLS].values
    X = []
    for i in range(0, len(data) - seq_len+1):
        X.append(data[i : i + seq_len, :])
    return np.array(X)

# ---------------------------------------------------------------------------
#  Helper: Add predictions to a DataFrame if missing
# ---------------------------------------------------------------------------
def _add_predictions_to_df(df_orig: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Añade una columna `pred` a un DataFrame crudo aplicando el
    modelo y escaladores correspondientes. Devuelve un nuevo
    DataFrame alineado con `date`, `real` (shift‑1) y `pred`.
    """
    # ------------------------------------------------------------------
    # 1. Ensure mandatory features exist – create dummies if absent
    # ------------------------------------------------------------------
    for col in FEATURE_COLS:
        if col not in df_orig.columns:
            df_orig[col] = 0.0

    # Handle Stock Splits / volume transform exactly like training
    df_orig["Stock_Split_Flag"] = (
        df_orig.get("Stock_Split_Flag")
        if "Stock_Split_Flag" in df_orig.columns
        else (df_orig.get("Stock Splits", 0) != 0).astype(int)
    )
    df_orig["Volume"] = np.log1p(df_orig["Volume"])

    # ------------------------------------------------------------------
    # 2. Load model & scalers
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS, ticker, "modelo.keras")
    scaler_path = os.path.join(MODELS, ticker, "scaler.save")
    close_scaler_path = os.path.join(MODELS, ticker, "close_scaler.save")

    for p in (model_path, scaler_path, close_scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    close_scaler = joblib.load(close_scaler_path)

    df_scaled = df_orig.copy()
    df_scaled[FEATURE_COLS[:-1]] = scaler.transform(df_scaled[FEATURE_COLS[:-1]])

    X = _create_sequences(df_scaled)
    if len(X) == 0:
        raise ValueError(
            "El archivo es demasiado pequeño para generar secuencias y predicciones."
        )

    preds_scaled = model.predict(X, verbose=0).reshape(-1, 1)
    preds_real = close_scaler.inverse_transform(preds_scaled).flatten()

    # --- alinear fecha y precio real (próximo día) --------------------------
    close_col = "Close" if "Close" in df_orig.columns else "real"
    # Shift hour +1, but wrap 16:30 to 09:30 next day
    if "ts_hour" in df_orig.columns:
        df_orig["ts_hour"] = (
            pd.to_datetime(df_orig["ts_hour"], errors="coerce")
            + pd.Timedelta(hours=1)
        ).apply(lambda ts: (ts + pd.Timedelta(hours=17)).replace(hour=9, minute=30)
                        if ts.hour == 16 and ts.minute == 30 else ts)
    elif "Date" in df_orig.columns:
        df_orig["Date"] = (
            pd.to_datetime(df_orig["Date"], errors="coerce")
            + pd.Timedelta(hours=1)
        ).apply(lambda ts: (ts + pd.Timedelta(hours=17)).replace(hour=9, minute=30)
                        if ts.hour == 16 and ts.minute == 30 else ts)

    date_col  = (
        "ts_hour" if "ts_hour" in df_orig.columns
        else ("Date" if "Date" in df_orig.columns else "date")
    )

    target_real  = df_orig[close_col].shift(-1).iloc[SEQ_LEN - 1 :].values
    target_dates = df_orig[date_col].iloc[SEQ_LEN-1:].values

    df_out = pd.DataFrame(
        {
            "date": target_dates,
            "real": target_real,
            "pred": preds_real,
        }
    )
    return df_out

def predict_uploaded_db(ticker: str, raw_csv_path: str) -> str:
    """
    Pipeline completo para un CSV subido por el usuario:
    limpia, escala, genera secuencias, predice, alinea y guarda
    un archivo `_predicted.csv` a disco. Devuelve la ruta resultante.
    """
    model_path = os.path.join(MODELS, ticker, "modelo.keras")
    scaler_path = os.path.join(MODELS, ticker, "scaler.save")
    close_scaler_path = os.path.join(MODELS, ticker, "close_scaler.save")

    for p in (model_path, scaler_path, close_scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # --- load artefacts
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    close_scaler = joblib.load(close_scaler_path)

    # --- read & basic cleaning
    df = pd.read_csv(raw_csv_path)
    df["Stock_Split_Flag"] = (df.get("Stock Splits", 0) != 0).astype(int)
    df["Volume"] = np.log1p(df["Volume"])
    df[FEATURE_COLS[:-1]] = scaler.transform(df[FEATURE_COLS[:-1]])

    X = _create_sequences(df)
    if len(X) == 0:
        raise ValueError("El archivo es demasiado pequeño para generar secuencias")

    preds_scaled = model.predict(X, verbose=0).reshape(-1, 1)
    preds_real = close_scaler.inverse_transform(preds_scaled).flatten()

    # Align real prices (Close) with predictions (the target was Next_Close)
    target_real = df["Close"].shift(-1).iloc[SEQ_LEN - 1 :].values

    out = pd.DataFrame(
        {
            "date": df["ts_hour"].iloc[SEQ_LEN - 1 :].values,
            "real": target_real,
            "pred": preds_real,
        }
    )

    # --- save result --------------------------------------------------------
    uploads_dir = os.path.join(MODELS, ticker, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    clean_name = os.path.basename(raw_csv_path).replace(".csv", "_predicted.csv")
    out_path = os.path.join(uploads_dir, clean_name)
    out.to_csv(out_path, index=False)

    return out_path

import yfinance as yf
from datetime import datetime

def generate_updated_db(ticker: str) -> str:
    """
    Descarga precios OHLCV de la última hora/mes con `yfinance`,
    guarda el CSV bruto y devuelve la ruta al archivo.
    """
    out_dir = os.path.join(MODELS, ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "db_updated_latest.csv")

    # Download last month hourly
    df = (
        yf.Ticker(ticker)
          .history(period="1mo", interval="60m")
          .reset_index()
          .rename(columns={"Datetime": "ts_hour"})
    )

    if df.empty:
        raise ValueError(f"yfinance devolvió 0 filas para {ticker}")

    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def download_and_predict_last_month(ticker: str) -> str:
    """
    Replica end‑to‑end el cuaderno de entrenamiento para un rango
    de un mes: descarga precios, analiza noticias, computa sentimiento,
    fusiona, predice y persiste el CSV final con columnas `date`,
    `real`, `pred`. Devuelve su ruta.
    """
    # ------------------------------------------------------------------ #
    # 1) Hourly prices
    # ------------------------------------------------------------------ #
    merged_dir = os.path.join(MODELS, ticker)
    os.makedirs(merged_dir, exist_ok=True)

    prices_csv = generate_updated_db(ticker)  # reuse helper
    df_prices = pd.read_csv(prices_csv, parse_dates=["ts_hour"])

    # ------------------------------------------------------------------ #
    # 2) Latest news & sentiment
    # ------------------------------------------------------------------ #
    UTC = pytz.UTC
    EASTERN = pytz.timezone("America/New_York")
    cutoff_utc = UTC.localize(datetime.utcnow() - timedelta(days=30))

    news_rows = []
    page = 1
    stop = False

    while not stop:
        url = f"https://markets.businessinsider.com/news/{ticker.lower()}-stock?p={page}"
        resp = requests.get(url, timeout=20)
        if not resp.ok:
            break

        # Try lxml (faster); if not available, fall back to built‑in parser
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except FeatureNotFound:
            soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.find_all("div", class_="latest-news__story")
        if not articles:
            break

        for art in articles:
            time_tag = art.find("time", class_="latest-news__date")
            if not time_tag:
                continue
            dt_raw = time_tag["datetime"]

            try:
                dt_utc = pd.to_datetime(dt_raw, format="%m/%d/%Y %I:%M:%S %p", utc=True)
            except ValueError:
                continue

            if dt_utc < cutoff_utc:
                stop = True
                break

            dt_et = dt_utc.tz_convert(EASTERN)
            title  = art.find("a", class_="news-link").get_text(strip=True)
            link   = art.find("a", class_="news-link")["href"]
            source = art.find("span", class_="latest-news__source").get_text(strip=True)

            news_rows.append([dt_raw, dt_et, title, source, link])

        page += 1

    if news_rows:
        df_news = pd.DataFrame(news_rows, columns=["datetime_raw","datetime_et","title","source","link"])
        df_news["datetime_et"] = pd.to_datetime(df_news["datetime_et"])

        # Keep only regular‑hours articles
        df_news = (
            df_news
            .set_index("datetime_et")
            .between_time("09:30", "15:30")
            .reset_index()
        )

        # Bucket to 1‑h windows starting 09:30
        def _bucket_1h(ts):
            market_open = ts.replace(hour=9, minute=30, second=0, microsecond=0)
            return market_open + pd.Timedelta(hours=int(((ts - market_open).total_seconds()) // 3600))

        df_news["ts_hour"] = df_news["datetime_et"].map(_bucket_1h)

        # FinBERT sentiment
        nlp = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=False,
            device=-1
        )
        results = nlp(df_news["title"].tolist(), batch_size=16)
        df_news["sentiment_label"] = [r["label"] for r in results]
        df_news["sentiment_score"] = [r["score"] for r in results]

        mapping = {"positive": 1, "neutral": 0, "negative": -1}
        df_news["label_val"] = df_news["sentiment_label"].map(mapping)
        df_news["weighted"] = df_news["sentiment_score"] * df_news["label_val"]

        df_sent = (
            df_news
            .groupby("ts_hour")
            .agg(
                cnt_articles=("weighted","count"),
                sum_weighted=("weighted","sum")
            )
            .reset_index()
        )
        df_sent["bucket_sentiment"] = df_sent["sum_weighted"] / df_sent["cnt_articles"]
    else:
        df_sent = pd.DataFrame(columns=["ts_hour","cnt_articles","sum_weighted","bucket_sentiment"])

    # ------------------------------------------------------------------ #
    # 3) Merge prices + sentiment
    # ------------------------------------------------------------------ #
    df_prices["ts_hour"] = pd.to_datetime(df_prices["ts_hour"])
    df_merged = df_prices.merge(df_sent, on="ts_hour", how="left")
    for col in ["cnt_articles","sum_weighted","bucket_sentiment"]:
        df_merged[col] = df_merged[col].fillna(0.0)

    merged_csv = os.path.join(merged_dir, "db_last_month.csv")
    df_merged.to_csv(merged_csv, index=False, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 4) Predictions
    # ------------------------------------------------------------------ #
    df_pred = _add_predictions_to_df(df_merged.copy(), ticker)
    pred_csv = os.path.join(merged_dir, "db_last_month_pred.csv")
    df_pred.to_csv(pred_csv, index=False, encoding="utf-8")

    return pred_csv

# ---------------------------------------------------------------------------
#  Public helper for API: return DataFrame date, real, pred (last‑month update)
# ---------------------------------------------------------------------------
def get_updated_month_prediction_df(ticker: str) -> pd.DataFrame:
    """
    Asegura que el CSV “último mes + predicciones” esté actualizado
    (lo regenera si es necesario) y lo devuelve como DataFrame
    listo para el front‑end, con métricas y `next_pred` adjuntos.
    """
    pred_csv = download_and_predict_last_month(ticker)  # generate / refresh
    df = pd.read_csv(pred_csv)

    # Normalise column names if necessary
    rename_map = {
        "ts_hour": "date",
        "Date": "date",
        "Close": "real",
        "Real": "real",
        "Pred_Close": "pred",
        "Pred": "pred",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["date"] = _to_et_series(df["date"])
    df = df[["date", "real", "pred"]]
    df = _add_metrics_and_next(df, ticker)
    return df