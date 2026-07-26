import json
import os
import random
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from .predictor import SpamPredictor
except ImportError:
    from predictor import SpamPredictor


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
METRICS_PATH = BASE_DIR.parent / "outputs_dl" / "metrics.json"

METRIC_INFO = {
    "accuracy": ("Accuracy", "(TP + TN) / (TP + TN + FP + FN)"),
    "precision": ("Precision", "TP / (TP + FP)"),
    "recall": ("Recall", "TP / (TP + FN)"),
    "f1": ("F1-Score", "2 × (Precision × Recall) / (Precision + Recall)"),
    "roc_auc": ("ROC-AUC", "Area under ROC curve across all classification thresholds"),
}

_predictor = None
_sample_cache: dict[str, list] | None = None
_metrics_cache: dict[str, float] | None = None


def get_predictor() -> SpamPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SpamPredictor()
    return _predictor


def _normalize_label_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    return numeric.fillna(text)


def _load_sample_cache() -> dict[str, list]:
    global _sample_cache
    if _sample_cache is not None:
        return _sample_cache

    cache: dict[str, list] = {"ham": [], "spam": []}

    if DATA_PATH.exists():
        try:
            df = pd.read_csv(
                DATA_PATH,
                usecols=["text", "label"],
                encoding="utf-8-sig",
                on_bad_lines="skip",
            )
            df = df.dropna(subset=["label", "text"]).copy()
            df["label"] = _normalize_label_column(df["label"])
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)

            for label_name, target in (("ham", 0), ("spam", 1)):
                cache[label_name] = df[df["label"] == target]["text"].astype(str).tolist()
        except Exception as e:
            app.logger.warning("Failed to load samples from CSV: %s", e)

    _sample_cache = cache
    return cache


def get_sample_from_csv(label: str) -> str:
    samples = _load_sample_cache().get(label, [])
    if not samples:
        return ""
    return random.choice(samples)


def get_metrics() -> dict[str, float]:
    global _metrics_cache
    if _metrics_cache is None:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                _metrics_cache = json.load(f)
        else:
            _metrics_cache = {}
    return _metrics_cache


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Email Spam Classification API is running.",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "sample_ham": "/sample?label=ham",
                "sample_spam": "/sample?label=spam",
                "metrics": "/metrics/<accuracy|precision|recall|f1|roc_auc>",
            },
        }
    )


@app.get("/health")
def health():
    try:
        predictor_ready = False

        try:
            get_predictor()
            predictor_ready = True
        except Exception:
            predictor_ready = False

        return jsonify(
            {
                "status": "ok",
                "service": "email-spam-classifier",
                "predictor_ready": predictor_ready,
                "dataset_ready": DATA_PATH.exists(),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}

        text = str(payload.get("text") or payload.get("email") or "").strip()

        if not text:
            return jsonify({"error": "Field 'text' or 'email' is required."}), 400

        result = get_predictor().predict(text)

        label = str(result.get("label", "ham"))
        probability = float(result.get("probability", 0.0))
        confidence = float(result.get("confidence", 0.0))

        return jsonify(
            {
                "prediction": label,
                "label": label,
                "probability": probability,
                "spam_probability": probability,
                "ham_probability": round(1.0 - probability, 6),
                "confidence": confidence,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/sample")
def sample():
    label = str(request.args.get("label") or "").strip().lower()

    if label not in ("ham", "spam"):
        return jsonify({"error": "Use /sample?label=ham or /sample?label=spam"}), 400

    text = get_sample_from_csv(label)

    if not text:
        return jsonify({"error": "Sample not available"}), 500

    return jsonify({"label": label, "text": text})


@app.get("/metrics/<key>")
def metrics(key: str):
    key = key.strip().lower()

    if key not in METRIC_INFO:
        return jsonify({"error": f"Unknown metric '{key}'. Use one of: {', '.join(METRIC_INFO)}"}), 400

    values = get_metrics()

    if key not in values:
        return jsonify({"error": "Metrics not available."}), 500

    label, formula = METRIC_INFO[key]
    return jsonify({"metric": label, "value": float(values[key]), "formula": formula})


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()