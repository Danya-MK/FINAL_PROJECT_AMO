import os
import json
import threading
from datetime import datetime

import random

import pandas as pd
from flask import Flask, request, jsonify
from flask import render_template_string

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = Flask(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "wine_quality_binary_model")
LOG_PATH = os.getenv("AB_LOG_PATH", "./logs/ab_requests.csv")

_lock = threading.Lock()
_b_traffic = float(os.getenv("AB_B_TRAFFIC", "0.3"))


def _get_stage_version(stage: str):
    client = MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=[stage])
    return latest[0].version if latest else None


def load_model_by_stage(stage: str):
    uri = f"models:/{MODEL_NAME}/{stage}"
    return mlflow.sklearn.load_model(uri)


def safe_load_models():
    # Production обязателен для A/B (если нет — сначала назначьте Production в MLflow UI)
    prod = load_model_by_stage("Production")
    try:
        stag = load_model_by_stage("Staging")
    except Exception:
        stag = prod
    return prod, stag


model_A, model_B = safe_load_models()
ver_A = _get_stage_version("Production")
ver_B = _get_stage_version("Staging")


def choose_variant(user_id: int | None):
    global _b_traffic
    if user_id is None:
        import time
        r = (time.time_ns() % 10_000) / 10_000
        return "B" if r < _b_traffic else "A"

    return "B" if (user_id % 100) < int(_b_traffic * 100) else "A"


def append_log(row: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    exists = os.path.exists(LOG_PATH)
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not exists, index=False)


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)

    user_id = payload.get("user_id")
    request_id = payload.get("request_id")
    features = payload.get("features", {})
    y_true = payload.get("y_true")

    variant = choose_variant(user_id)
    stage = "Production" if variant == "A" else "Staging"
    model = model_A if variant == "A" else model_B
    model_version = ver_A if variant == "A" else ver_B

    X = pd.DataFrame([features])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[:, 1][0])
        pred_label = int(proba >= 0.5)
    else:
        pred_label = int(model.predict(X)[0])

    row = {
        "ts": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "user_id": user_id,
        "variant": variant,
        "stage": stage,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "features_json": json.dumps(features, ensure_ascii=False),
        "prediction_label": pred_label,
        "prediction_proba": proba,
        "y_true": y_true
    }
    append_log(row)

    return jsonify({
        "variant": variant,
        "stage": stage,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "prediction_label": pred_label,
        "prediction_proba": proba
    })


@app.get("/config")
def get_config():
    return jsonify({"b_traffic": _b_traffic, "model_name": MODEL_NAME})


@app.post("/config/traffic")
def set_traffic():
    global _b_traffic
    body = request.get_json(force=True)
    val = float(body["b_traffic"])
    if not (0.0 <= val <= 1.0):
        return jsonify({"error": "b_traffic must be in [0, 1]"}), 400
    with _lock:
        _b_traffic = val
    return jsonify({"b_traffic": _b_traffic})


@app.post("/reload_models")
def reload_models():
    global model_A, model_B, ver_A, ver_B
    model_A, model_B = safe_load_models()
    ver_A = _get_stage_version("Production")
    ver_B = _get_stage_version("Staging")
    return jsonify({"status": "reloaded", "ver_A": ver_A, "ver_B": ver_B})

@app.get("/")
def home():
    return jsonify({"status": "ok", "service": "ab_flask", "model_name": MODEL_NAME})

DATA_PATH = os.getenv("AB_DATA_PATH", "/app/data/current.csv")

_cached_df = None

def _load_current_df():
    global _cached_df
    if _cached_df is None:
        _cached_df = pd.read_csv(DATA_PATH)
    return _cached_df

@app.get("/sample")
def sample():
    """
    Возвращает случайную строку из current.csv:
    { y_true: 0/1, features: {...} }
    """
    df = _load_current_df()
    if df.empty:
        return jsonify({"error": "current.csv is empty"}), 400

    row = df.sample(1).iloc[0].to_dict()
    y_true = int(row["target"])
    features = {k: row[k] for k in row.keys() if k != "target"}

    return jsonify({"y_true": y_true, "features": features})

@app.get("/ui")
def ui():
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>A/B Predict UI</title>
        <style>
          body { font-family: Arial; margin: 24px; max-width: 900px; }
          label { display:block; margin-top:10px; }
          input { width: 280px; padding: 6px; }
          pre { background:#f4f4f4; padding:12px; }
          .row { display:flex; gap:24px; flex-wrap:wrap; }
          .card { border:1px solid #ddd; padding:16px; border-radius:8px; }
          button { margin-top: 12px; padding: 8px 14px; }
        </style>
      </head>
      <body>
        <h2>Flask A/B Predict UI</h2>
        <div class="row">
          <div class="card">
            <h3>Request</h3>
            <label>request_id <input id="request_id" value="req-1"></label>
            <label>user_id <input id="user_id" value="123"></label>
            <label>y_true (0/1) <input id="y_true" value="1"></label>

            <h4>Features (Wine Quality)</h4>
            <label>fixed acidity <input id="fixed_acidity" value="7.4"></label>
            <label>volatile acidity <input id="volatile_acidity" value="0.49"></label>
            <label>citric acid <input id="citric_acid" value="0.27"></label>
            <label>residual sugar <input id="residual_sugar" value="2.1"></label>
            <label>chlorides <input id="chlorides" value="0.071"></label>
            <label>free sulfur dioxide <input id="free_sulfur_dioxide" value="14.0"></label>
            <label>total sulfur dioxide <input id="total_sulfur_dioxide" value="25.0"></label>
            <label>density <input id="density" value="1.034"></label>
            <label>pH <input id="pH" value="3.25"></label>
            <label>sulphates <input id="sulphates" value="0.63"></label>
            <label>alcohol <input id="alcohol" value="13.0"></label>

            <button onclick="sendPredict()">Send /predict</button>
            <button onclick="loadSample()">Load random sample</button>
          </div>

          <div class="card">
            <h3>A/B traffic</h3>
            <label>b_traffic (0..1) <input id="b_traffic" value="0.3"></label>
            <button onclick="setTraffic()">Set traffic</button>
            <button onclick="reloadModels()">Reload models</button>

            <h3>Response</h3>
            <pre id="out">{}</pre>
          </div>
        </div>

        <script>
        async function loadSample() {
            const r = await fetch("/sample");
            const j = await r.json();

            if (j.error) {
            document.getElementById("out").textContent = JSON.stringify(j, null, 2);
            return;
            }

            document.getElementById("y_true").value = j.y_true;

            document.getElementById("fixed_acidity").value = j.features["fixed acidity"];
            document.getElementById("volatile_acidity").value = j.features["volatile acidity"];
            document.getElementById("citric_acid").value = j.features["citric acid"];
            document.getElementById("residual_sugar").value = j.features["residual sugar"];
            document.getElementById("chlorides").value = j.features["chlorides"];
            document.getElementById("free_sulfur_dioxide").value = j.features["free sulfur dioxide"];
            document.getElementById("total_sulfur_dioxide").value = j.features["total sulfur dioxide"];
            document.getElementById("density").value = j.features["density"];
            document.getElementById("pH").value = j.features["pH"];
            document.getElementById("sulphates").value = j.features["sulphates"];
            document.getElementById("alcohol").value = j.features["alcohol"];

            document.getElementById("out").textContent = JSON.stringify(j, null, 2);
        }

        async function sendPredict() {
            const payload = {
            request_id: document.getElementById("request_id").value,
            user_id: parseInt(document.getElementById("user_id").value),
            y_true: parseInt(document.getElementById("y_true").value),
            features: {
                "fixed acidity": parseFloat(document.getElementById("fixed_acidity").value),
                "volatile acidity": parseFloat(document.getElementById("volatile_acidity").value),
                "citric acid": parseFloat(document.getElementById("citric_acid").value),
                "residual sugar": parseFloat(document.getElementById("residual_sugar").value),
                "chlorides": parseFloat(document.getElementById("chlorides").value),
                "free sulfur dioxide": parseFloat(document.getElementById("free_sulfur_dioxide").value),
                "total sulfur dioxide": parseFloat(document.getElementById("total_sulfur_dioxide").value),
                "density": parseFloat(document.getElementById("density").value),
                "pH": parseFloat(document.getElementById("pH").value),
                "sulphates": parseFloat(document.getElementById("sulphates").value),
                "alcohol": parseFloat(document.getElementById("alcohol").value),
            }
            };

            const r = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify(payload)
            });

            const j = await r.json();
            document.getElementById("out").textContent = JSON.stringify(j, null, 2);
        }

        async function setTraffic() {
            const val = parseFloat(document.getElementById("b_traffic").value);
            const r = await fetch("/config/traffic", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({b_traffic: val})
            });
            document.getElementById("out").textContent = JSON.stringify(await r.json(), null, 2);
        }

        async function reloadModels() {
            const r = await fetch("/reload_models", {method:"POST"});
            document.getElementById("out").textContent = JSON.stringify(await r.json(), null, 2);
        }
        </script>
      </body>
    </html>
    """
    return render_template_string(html)