import os
import re
import joblib
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.joblib")
FEATURE_MASK_PATH = os.path.join(ARTIFACT_DIR, "feature_mask.npy")
SVD_PATH = os.path.join(ARTIFACT_DIR, "svd.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.keras")


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = s.replace("\n", " ").replace("\r", " ")

    # normalize repeats of escapenumber -> <NUM>
    s = re.sub(r"\b(?:escapenumber\b(?:\s+|$))+",
               " <NUM> ",
               s)

    # normalize actual numbers -> <NUM>
    s = re.sub(r"\b\d+\b", " <NUM> ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


class SpamPredictor:
    def __init__(self):
        self.vectorizer = None
        self.feature_mask = None
        self.svd = None
        self.model = None

    def load(self):
        required = [VECTORIZER_PATH, FEATURE_MASK_PATH, SVD_PATH, MODEL_PATH]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing model artifacts. Train first (python backend/train.py).\nMissing:\n"
                + "\n".join(missing)
            )

        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.feature_mask = np.load(FEATURE_MASK_PATH)
        self.svd = joblib.load(SVD_PATH)
        self.model = keras.models.load_model(MODEL_PATH)

    def predict_proba(self, text: str) -> float:
        if self.model is None:
            self.load()

        text = clean_text(text)
        X = self.vectorizer.transform([text])          # sparse
        X_fs = X[:, self.feature_mask]                 # sparse
        X_dense = self.svd.transform(X_fs)             # dense
        p_spam = float(self.model.predict(X_dense, verbose=0).ravel()[0])
        return p_spam

    def predict(self, text: str):
        p = self.predict_proba(text)
        label = 1 if p >= 0.5 else 0
        return label, p