import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow import keras


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "baseline_results")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

CONF_MATRIX_PATH = os.path.join(ARTIFACT_DIR, "baseline_confusion_matrix.csv")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "baseline_classification_report.json")

TFIDF_MAX_FEATURES = 50000
SVD_COMPONENTS = 250
BATCH_SIZE = 64
EPOCHS = 15


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\b(?:escapenumber\b(?:\s+|$))+", " <NUM> ", s)
    s = re.sub(r"\b\d+\b", " <NUM> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def train():
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

    # TF-IDF (NO FEATURE SELECTION)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    # SVD directly
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    X_train_dense = svd.fit_transform(X_train)
    X_test_dense = svd.transform(X_test)

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

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    print("\nBaseline Confusion Matrix:\n", cm)
    print("\nBaseline Classification Report:\n",
          classification_report(y_test, preds))

    # Save results
    cm_df = pd.DataFrame(cm,
                         columns=["Pred_Ham", "Pred_Spam"],
                         index=["True_Ham", "True_Spam"])
    cm_df.to_csv(CONF_MATRIX_PATH)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Baseline results saved to:", ARTIFACT_DIR)


if __name__ == "__main__":
    train()