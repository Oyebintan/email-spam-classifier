import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow import keras


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model")

VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.joblib")
FILTER_SELECTOR_PATH = os.path.join(ARTIFACT_DIR, "filter_selector.joblib")
EMBED_SELECTOR_PATH = os.path.join(ARTIFACT_DIR, "embed_selector.joblib")
FEATURE_MASK_PATH = os.path.join(ARTIFACT_DIR, "feature_mask.npy")
SVD_PATH = os.path.join(ARTIFACT_DIR, "svd.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.keras")

CONF_MATRIX_PATH = os.path.join(ARTIFACT_DIR, "confusion_matrix.csv")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "classification_report.json")


TFIDF_MAX_FEATURES = 50000
CHI2_KBEST = 15000
SVD_COMPONENTS = 250
BATCH_SIZE = 64
EPOCHS = 15


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = s.replace("\n", " ").replace("\r", " ")

    s = re.sub(r"\b(?:escapenumber\b(?:\s+|$))+",
               " <NUM> ",
               s)

    s = re.sub(r"\b\d+\b", " <NUM> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_artifacts_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def build_hybrid_mask(X_sparse, y, k_best: int, l1_c: float = 1.0):
    k_best = min(k_best, X_sparse.shape[1])

    filter_sel = SelectKBest(score_func=chi2, k=k_best)
    X_f = filter_sel.fit_transform(X_sparse, y)

    lr = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=3000,
        C=l1_c,
        random_state=42
    )
    lr.fit(X_f, y)

    feature_mask = filter_sel.get_support(indices=False)
    return filter_sel, lr, feature_mask


def train():
    ensure_artifacts_dir()

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)

    y = df["label"].values
    X_text = df["text"].values

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    filter_sel, embed_lr, feature_mask = build_hybrid_mask(
        X_train, y_train, k_best=CHI2_KBEST
    )

    X_train_fs = X_train[:, feature_mask]
    X_test_fs = X_test[:, feature_mask]

    n_comp = min(SVD_COMPONENTS, X_train_fs.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)

    X_train_dense = svd.fit_transform(X_train_fs)
    X_test_dense = svd.transform(X_test_fs)

    tf.random.set_seed(42)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_dense.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.35),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_dense, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    probs = model.predict(X_test_dense, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    # ===== Evaluation =====
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(y_test, preds))

    # ===== Export Results =====

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm,
                         columns=["Pred_Ham", "Pred_Spam"],
                         index=["True_Ham", "True_Spam"])
    cm_df.to_csv(CONF_MATRIX_PATH)

    # Save classification report as JSON
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Metrics exported to:")
    print(CONF_MATRIX_PATH)
    print(REPORT_PATH)

    # ===== Save Model Artifacts =====

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(filter_sel, FILTER_SELECTOR_PATH)
    joblib.dump(embed_lr, EMBED_SELECTOR_PATH)
    np.save(FEATURE_MASK_PATH, feature_mask)
    joblib.dump(svd, SVD_PATH)
    model.save(MODEL_PATH)

    print("\n✅ Training complete.")
    print("Artifacts saved to:", ARTIFACT_DIR)


if __name__ == "__main__":
    train()