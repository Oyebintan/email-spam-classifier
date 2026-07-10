"""Hybrid Feature Selection + Deep Learning for Email Spam Classification.

Pipeline:
1) Text preprocessing with TF-IDF
2) Filter selection using Chi-square
3) Embedded selection using L1 Logistic Regression
4) Final classification using TensorFlow/Keras deep neural network
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def build_feature_pipeline(max_features: int, chi2_k: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=max_features,
                ),
            ),
            ("chi2", SelectKBest(score_func=chi2, k=chi2_k)),
        ]
    )


def build_l1_selector(c: float, random_state: int) -> SelectFromModel:
    # L1 logistic regression used only for embedded feature selection
    estimator = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=c,
        max_iter=2000,
        random_state=random_state,
    )
    return SelectFromModel(estimator=estimator, threshold=1e-8)


def normalize_labels(series: pd.Series) -> np.ndarray:
    raw = series.astype(str).str.strip().str.lower()
    mapped = raw.replace(
        {
            "spam": "1",
            "ham": "0",
            "true": "1",
            "false": "0",
        }
    )
    return mapped.values


def build_deep_model(input_dim: int, hidden_dims: list[int], dropout: float) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation="relu"))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Metrics:
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name.")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name.")
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--chi2_k", type=int, default=5000)
    parser.add_argument("--l1_c", type=float, default=1.0)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_dl")
    args = parser.parse_args()

    tf.random.set_seed(args.random_state)
    np.random.seed(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"Dataset must include columns '{args.text_col}' and '{args.label_col}'."
        )

    df = df[[args.text_col, args.label_col]].dropna().drop_duplicates()

    x_raw = df[args.text_col].astype(str).values
    y_raw = normalize_labels(df[args.label_col])

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    feature_pipeline = build_feature_pipeline(
        max_features=args.max_features,
        chi2_k=args.chi2_k,
    )

    x_train_chi2 = feature_pipeline.fit_transform(x_train_raw, y_train)
    x_test_chi2 = feature_pipeline.transform(x_test_raw)

    if hasattr(x_train_chi2, "toarray"):
        x_train_chi2 = x_train_chi2.toarray().astype(np.float32)
        x_test_chi2 = x_test_chi2.toarray().astype(np.float32)
    else:
        x_train_chi2 = np.asarray(x_train_chi2, dtype=np.float32)
        x_test_chi2 = np.asarray(x_test_chi2, dtype=np.float32)

    l1_selector = build_l1_selector(c=args.l1_c, random_state=args.random_state)
    x_train_selected = l1_selector.fit_transform(x_train_chi2, y_train)
    x_test_selected = l1_selector.transform(x_test_chi2)

    # safety fallback if very few features survive
    if x_train_selected.shape[1] == 0:
        raise ValueError("L1 feature selection removed all features. Increase --l1_c.")
    duplicate_single_feature = x_train_selected.shape[1] == 1
    if duplicate_single_feature:
        x_train_selected = np.hstack([x_train_selected, x_train_selected])
        x_test_selected = np.hstack([x_test_selected, x_test_selected])

    x_train_selected = x_train_selected.astype(np.float32)
    x_test_selected = x_test_selected.astype(np.float32)

    model = build_deep_model(
        input_dim=x_train_selected.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train_selected,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    y_proba = model.predict(x_test_selected, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = evaluate(y_test, y_pred, y_proba)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    np.save(output_dir / "confusion_matrix.npy", cm)

    # save preprocessing + feature selection objects
    artifact = {
        "feature_pipeline": feature_pipeline,
        "l1_selector": l1_selector,
        "label_encoder": encoder,
        "duplicate_single_feature": duplicate_single_feature,
    }
    joblib.dump(artifact, output_dir / "pipeline.pkl")

    # save deep learning model only
    model.save(output_dir / "model.keras")

    print("=== Training Complete ===")
    print("Model backend: tensorflow")
    print(f"Selected feature count: {x_train_selected.shape[1]}")
    print(json.dumps(asdict(metrics), indent=2))
    print(report)


if __name__ == "__main__":
    main()