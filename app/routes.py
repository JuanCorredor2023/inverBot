"""
routes.py — Rutas Flask para InverBot
=====================================
Este módulo define la interfaz HTTP (API + páginas) del proyecto.
Incluye endpoints de:
  * Páginas estáticas (`/`, `/about_us`, `/demo`, `/dashboard`)
  * API REST para predicciones (`/api/predict/*`)
  * Descarga de modelos y bases de datos
  * Carga de CSVs para predicción en línea

Todas las funciones tienen docstrings en español para facilitar el
mantenimiento y la colaboración futura.
"""
from flask import Blueprint, render_template, jsonify, send_from_directory, request
from .services import (
    get_prediction_df,
    list_models,
    predict_uploaded_db,
    download_and_predict_last_month,
    get_updated_month_prediction_df,   # NEW
)
import math
def _safe_records(df):
    """
    Convert DataFrame to list‑of‑dicts replacing NaN/inf with None
    so that jsonify can serialize valid JSON.
    """
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None
    return records
import os
from werkzeug.utils import secure_filename
import uuid
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Renderiza la página principal de la aplicación."""
    return render_template('index.html')

@main.route('/about_us')
def about_us():
    """Muestra la página «Acerca de nosotros» con información del equipo."""
    return render_template('about_us.html')

@main.route('/demo')
def demo():
    """Renderiza la página de demostración interactiva."""
    return render_template('demo.html')

@main.route('/dashboard')
def dashboard():
    """
    Renderiza el panel de análisis predictivo con los selectores de ticker
    y dataset, inyectando la lista de tickers disponibles desde
    `list_models()`.
    """
    tickers = list_models()  # ['GOOG','AAPL',...]
    return render_template('dashboard.html', tickers=tickers)

@main.route('/simulation')
def simulation():
    """
    Renderiza el panel de análisis predictivo con los selectores de ticker
    y dataset, inyectando la lista de tickers disponibles desde
    `list_models()`.
    """
    tickers = list_models()  # ['GOOG','AAPL',...]
    return render_template('simulacion.html', tickers=tickers)

# Descarga del archivo .keras
@main.route('/download/<ticker>')
def download_model(ticker):
    """Descarga el archivo `modelo.keras` entrenado para el ticker dado."""
    path = os.path.join('modelos', ticker)
    return send_from_directory(path, 'modelo.keras', as_attachment=True)

@main.route('/api/models')
def api_models():
    """Devuelve la lista de tickers disponibles en formato JSON."""
    return jsonify(list_models())

@main.route('/api/predict/<ticker>/<ds>')
def api_predict(ticker, ds):
    """
    API REST que entrega precios reales, predicciones, métricas y la próxima
    predicción horaria para `<ticker>` y el dataset solicitado (`2y` o `1mo`).
    """
    df = get_prediction_df(ticker, ds)  #  returns DataFrame(date, real, pred)
    return jsonify({
        "data": _safe_records(df),
        "metrics": df.attrs.get("metrics", {}),
        "next_pred": df.attrs.get("next_pred")
    })

# -----------------------------------------------
#   Latest‑month on‑the‑fly prediction endpoint
# -----------------------------------------------
@main.route('/api/predict-updated/<ticker>')
def api_predict_updated(ticker):
    """
    Genera/actualiza la base del último mes para el `<ticker>` y devuelve
    JSON con precios, predicciones y métricas.
    """
    df = get_updated_month_prediction_df(ticker)
    return jsonify({
        "data": _safe_records(df),
        "metrics": df.attrs.get("metrics", {}),
        "next_pred": df.attrs.get("next_pred")
    })

@main.route('/download/db/<ticker>/<ds>')
def download_db(ticker, ds):
    """Descarga el CSV estático de 2 años o 1 mes no visto para el ticker."""
    filename = 'dbFinal_2y.csv' if ds=='2y' else 'db_NOSeen_1mo.csv'
    dir_ = os.path.join(os.path.dirname(__file__), 'modelos', ticker)
    return send_from_directory(dir_, filename, as_attachment=True)

# -----------------------------------------------------------
#  Upload a brand‑new CSV → run model → return JSON + clean URL
# -----------------------------------------------------------
@main.route("/upload/db/<ticker>", methods=["POST"])
def upload_db(ticker):
    """
    Recibe un CSV cargado por el usuario, ejecuta el modelo correspondiente
    y devuelve los datos predichos junto con la URL de descarga del archivo
    limpio.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{f.filename}")
    tmp_path = os.path.join("tmp", filename)
    os.makedirs("tmp", exist_ok=True)
    f.save(tmp_path)

    try:
        clean_path = predict_uploaded_db(ticker, tmp_path)
        df = pd.read_csv(clean_path)
        clean_url = f"/download/clean-db/{ticker}/{os.path.basename(clean_path)}"
        return jsonify(
            {"data": df.to_dict(orient="records"), "clean_url": clean_url}
        )
    finally:
        # keep the uploaded file if you need audit; else, uncomment next line
        # os.remove(tmp_path)
        pass


@main.route('/download/clean-db/<ticker>/<filename>')
def download_clean_db(ticker, filename):
    """Devuelve el CSV ya procesado y predicho que solicitó el usuario."""
    dir_ = os.path.join(os.path.dirname(__file__), 'modelos', ticker, 'uploads')
    return send_from_directory(dir_, filename, as_attachment=True)

@main.route('/download/latest-db')
def download_latest_db():
    """Descarga un ZIP con la base de datos global más actualizada."""
    path = os.path.join(os.path.dirname(__file__), 'latest_db.zip')  # genera según tu lógica
    return send_from_directory(os.path.dirname(path), os.path.basename(path), as_attachment=True)

# -----------------------------------------------------------
#  Download freshly‑generated 1‑month DB (prices + sentiment + preds)
# -----------------------------------------------------------
@main.route('/download/updated-db/<ticker>')
def download_updated_db(ticker):
    """
    Descarga (o genera si no existe) el CSV de 1 mes + predicciones para
    el ticker y lo envía como attachment.
    """
    csv_path = download_and_predict_last_month(ticker)  # crea y predice
    dir_, filename = os.path.split(csv_path)
    return send_from_directory(dir_, filename, as_attachment=True)


# -----------------------------------------------
#   Simulation-specific endpoint (shift +1)
# -----------------------------------------------
@main.route('/api/predict-sim/<ticker>/<ds>')
def api_predict_sim(ticker, ds):
    """
    Endpoint exclusivo para la página de simulación.
    Adelanta la serie REAL una hora (shift +1) de modo que
    real(t) y pred(t) se alineen en la misma fila.
    """
    df = get_prediction_df(ticker, ds)

    # Shift +1: la real de la hora (t-1) pasa a la fila t
    df['real'] = df['real'].shift(1)
    df = df.iloc[1:].reset_index(drop=True)   # descarta la primera (NaN)

    return jsonify({
        "data": _safe_records(df),
        "metrics": df.attrs.get("metrics", {}),
        "next_pred": df.attrs.get("next_pred")
    })