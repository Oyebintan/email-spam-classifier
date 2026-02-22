import re
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split


def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_sms_lines(path: str) -> pd.DataFrame:
    """
    Your dataset rows look like:  "ham\\tmessage..."  (tab inside quotes)
    So we read line-by-line and split on the first tab.
    """
    labels = []
    texts = []

    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # remove surrounding quotes if present
            if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
                line = line[1:-1]

            # split into label and text
            if "\t" in line:
                label, text = line.split("\t", 1)
            elif "," in line and line.lower().startswith(("ham,", "spam,")):
                # fallback in case someone converts it to comma format
                label, text = line.split(",", 1)
            else:
                continue

            label = label.strip().lower()
            if label not in ("ham", "spam"):
                continue

            labels.append(label)
            texts.append(text)

    df = pd.DataFrame({"label": labels, "text": texts})
    return df


def load_sms_dataset(csv_path: str) -> Tuple[pd.Series, np.ndarray]:
    """
    Loads SMS dataset robustly:
    - Handles normal Kaggle variants (v1/v2 or label/text)
    - Handles YOUR file format: "ham\\tmessage" (tab inside quotes)
    """
    df = None

    # Try normal CSV reading first
    try:
        df_try = pd.read_csv(csv_path, encoding="latin-1")
        if ("v1" in df_try.columns and "v2" in df_try.columns) or (
            "label" in df_try.columns and "text" in df_try.columns
        ):
            df = df_try
        else:
            # if it comes in as one weird column, it's your quoted-tab format
            df = None
    except Exception:
        df = None

    # Normalize normal CSV formats
    if df is not None:
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        elif "label" in df.columns and "text" in df.columns:
            df = df[["label", "text"]]

        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df = df[df["label"].isin(["ham", "spam"])].copy()
        df["text"] = df["text"].astype(str).map(basic_clean)

    # Your file format (quoted lines with tab inside)
    if df is None:
        df = _parse_sms_lines(csv_path)
        df["text"] = df["text"].astype(str).map(basic_clean)

    # labels
    y = (df["label"] == "spam").astype(int).to_numpy()

    # optional sanity output (helps during debugging)
    # print("[INFO] Label counts:", df["label"].value_counts().to_dict())

    return df["text"], y


@dataclass
class HybridFSArtifacts:
    vectorizer: TfidfVectorizer
    chi2_selector: SelectKBest
    rfe_selector: Optional[RFE]


def hybrid_feature_select_fit_transform(
    texts: pd.Series,
    y: np.ndarray,
    *,
    max_features_tfidf: int = 15000,
    chi2_k: int = 3000,
    rfe_k: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HybridFSArtifacts]:

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features_tfidf,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Chi2 filter
    chi2_k = min(chi2_k, X_train_tfidf.shape[1])
    chi2_selector = SelectKBest(score_func=chi2, k=chi2_k)
    X_train_chi2 = chi2_selector.fit_transform(X_train_tfidf, y_train)
    X_test_chi2 = chi2_selector.transform(X_test_tfidf)

    # If Chi2 produced too few features, skip RFE
    if X_train_chi2.shape[1] < 2:
        artifacts = HybridFSArtifacts(
            vectorizer=vectorizer,
            chi2_selector=chi2_selector,
            rfe_selector=None,
        )
        return (
            X_train_chi2.astype(np.float32).toarray(),
            X_test_chi2.astype(np.float32).toarray(),
            y_train,
            y_test,
            artifacts,
        )

    # RFE wrapper (use saga; robust)
    rfe_k = min(rfe_k, X_train_chi2.shape[1])
    base_estimator = LogisticRegression(max_iter=3000, solver="saga")
    rfe_selector = RFE(
        estimator=base_estimator,
        n_features_to_select=rfe_k,
        step=0.1,
    )

    X_train_sel = rfe_selector.fit_transform(X_train_chi2, y_train)
    X_test_sel = rfe_selector.transform(X_test_chi2)

    artifacts = HybridFSArtifacts(
        vectorizer=vectorizer,
        chi2_selector=chi2_selector,
        rfe_selector=rfe_selector,
    )

    return (
        X_train_sel.astype(np.float32).toarray(),
        X_test_sel.astype(np.float32).toarray(),
        y_train,
        y_test,
        artifacts,
    )


def hybrid_feature_select_transform(texts: pd.Series, artifacts: HybridFSArtifacts) -> np.ndarray:
    cleaned = texts.astype(str).map(basic_clean)

    X = artifacts.vectorizer.transform(cleaned)
    X = artifacts.chi2_selector.transform(X)

    if artifacts.rfe_selector is not None:
        X = artifacts.rfe_selector.transform(X)

    return X.astype(np.float32).toarray()