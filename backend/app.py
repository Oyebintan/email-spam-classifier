from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import secrets


from .predictor import SpamPredictor

app = Flask(__name__)
CORS(app)

predictor = SpamPredictor()

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")

df = None
ham_texts = []
spam_texts = []
_last_sample = {"ham": None, "spam": None}


def load_dataset():
    global df, ham_texts, spam_texts

    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Dataset not found at {DATA_PATH} (sample endpoint will not work).")
        return

    df = pd.read_csv(DATA_PATH).dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(int)

    ham_texts = df[df["label"] == 0]["text"].astype(str).tolist()
    spam_texts = df[df["label"] == 1]["text"].astype(str).tolist()

    print(f"✅ Dataset loaded: ham={len(ham_texts)}, spam={len(spam_texts)}")


def pick_random(label: str) -> str:
    pool = ham_texts if label == "ham" else spam_texts
    if not pool:
        return ""

    last = _last_sample[label]
    for _ in range(8):
        idx = secrets.randbelow(len(pool))
        text = pool[idx]
        if text != last:
            _last_sample[label] = text
            return text

    text = pool[secrets.randbelow(len(pool))]
    _last_sample[label] = text
    return text


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "email-spam-classifier"})


@app.get("/sample")
def sample():
    label = (request.args.get("label") or "").strip().lower()
    if label not in ("ham", "spam"):
        return jsonify({"error": "Use /sample?label=ham or /sample?label=spam"}), 400

    if df is None:
        return jsonify({"error": "Dataset not loaded. Ensure backend/spam.csv exists."}), 500

    text = pick_random(label)
    if not text:
        return jsonify({"error": f"No samples available for {label}"}), 500

    return jsonify({"label": label, "text": text})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Provide email text in JSON: {\"text\": \"...\"}"}), 400

    label, p_spam = predictor.predict(text)

    return jsonify({
        "label": "spam" if label == 1 else "ham",
        "spam_probability": round(p_spam, 6),
        "ham_probability": round(1.0 - p_spam, 6),
        "threshold": 0.5
    })


def main():
    load_dataset()
    app.run(host="127.0.0.1", port=8000, debug=True)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)