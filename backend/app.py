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

_predictor = None


def get_predictor() -> SpamPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SpamPredictor()
    return _predictor


def get_sample_from_csv(label: str, max_rows: int = 2000) -> str:
    if not DATA_PATH.exists():
        return ""

    try:
        df = pd.read_csv(DATA_PATH, nrows=max_rows, encoding="utf-8-sig", on_bad_lines="skip")
        print("Columns found:", df.columns.tolist())  # ← shows column names in logs
        print("Label values:", df["label"].unique()[:10])  # ← shows label values
        df = df.dropna(subset=["label", "text"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        target = 1 if label == "spam" else 0
        samples = df[df["label"] == target]["text"].astype(str).tolist()
        print(f"Found {len(samples)} samples for label {target}")  # ← shows count

        if not samples:
            return ""

        return random.choice(samples)
    except Exception as e:
        print(f"CSV ERROR: {e}")  # ← prints real error
        return ""



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

    text = get_sample_from_csv(label, max_rows=2000)

    if not text:
        return jsonify({"error": "Sample not available"}), 500

    return jsonify({"label": label, "text": text})


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()