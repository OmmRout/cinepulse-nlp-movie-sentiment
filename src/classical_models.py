"""Layer 5: Modeling Layer — Classical ML Baselines (Naive Bayes, Logistic Regression, Linear SVM)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE


def train_classical_models(
    matrices: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Train BoW + NB, TF-IDF + NB, TF-IDF + Logistic Regression, and TF-IDF + Linear SVM."""
    model_specs = [
        {
            "key": "bow_nb",
            "name": "BoW + Multinomial NB",
            "representation": "Bag-of-Words (1-2 grams)",
            "estimator": MultinomialNB(alpha=0.5),
            "X_train": matrices["X_train_bow"],
            "X_val": matrices["X_val_bow"],
            "X_test": matrices["X_test_bow"],
            "feature_space": "bow",
        },
        {
            "key": "tfidf_nb",
            "name": "TF-IDF + Multinomial NB",
            "representation": "TF-IDF (1-2 grams, sublinear)",
            "estimator": MultinomialNB(alpha=0.4),
            "X_train": matrices["X_train_tfidf"],
            "X_val": matrices["X_val_tfidf"],
            "X_test": matrices["X_test_tfidf"],
            "feature_space": "tfidf",
        },
        {
            "key": "tfidf_logreg",
            "name": "TF-IDF + Logistic Regression",
            "representation": "TF-IDF (1-2 grams, sublinear)",
            "estimator": LogisticRegression(
                C=2.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "X_train": matrices["X_train_tfidf"],
            "X_val": matrices["X_val_tfidf"],
            "X_test": matrices["X_test_tfidf"],
            "feature_space": "tfidf",
        },
        {
            "key": "tfidf_svm",
            "name": "TF-IDF + Linear SVM",
            "representation": "TF-IDF (1-2 grams, sublinear)",
            "estimator": CalibratedClassifierCV(
                LinearSVC(C=0.8, random_state=RANDOM_STATE),
                cv=3,
                method="sigmoid",
            ),
            "X_train": matrices["X_train_tfidf"],
            "X_val": matrices["X_val_tfidf"],
            "X_test": matrices["X_test_tfidf"],
            "feature_space": "tfidf",
        },
        {
            "key": "svd_logreg",
            "name": "Dense SVD Embeddings + LogReg",
            "representation": "LSA Dense Embedding (64-dim)",
            "estimator": LogisticRegression(
                C=1.5,
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "X_train": matrices["X_train_dense"],
            "X_val": matrices["X_val_dense"],
            "X_test": matrices["X_test_dense"],
            "feature_space": "dense",
        },
    ]

    trained: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        t0 = time.perf_counter()
        estimator = spec["estimator"]
        estimator.fit(spec["X_train"], y_train)
        train_ms = (time.perf_counter() - t0) * 1000.0

        val_acc = float(estimator.score(spec["X_val"], y_val))
        trained[spec["key"]] = {
            "key": spec["key"],
            "name": spec["name"],
            "representation": spec["representation"],
            "feature_space": spec["feature_space"],
            "estimator": estimator,
            "train_time_ms": round(train_ms, 2),
            "val_accuracy": round(val_acc, 4),
        }

    return trained


def extract_top_ngrams(
    tfidf_vectorizer,
    logreg_estimator: LogisticRegression,
    top_k: int = 18,
) -> dict[str, list[dict[str, float | str]]]:
    """Extract highest-weight positive and negative n-grams from TF-IDF Logistic Regression."""
    feature_names = np.asarray(tfidf_vectorizer.get_feature_names_out())
    coefs = logreg_estimator.coef_[0]

    top_pos_idx = np.argsort(coefs)[-top_k:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_k]

    return {
        "positive": [
            {"term": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
            for i in top_pos_idx
        ],
        "negative": [
            {"term": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
            for i in top_neg_idx
        ],
    }


def explain_single_review_tokens(
    cleaned_text: str,
    tfidf_vectorizer,
    logreg_estimator: LogisticRegression,
    top_n: int = 8,
) -> list[dict[str, float | str]]:
    """Compute per-token sentiment attribution for an unseen review using TF-IDF * LogReg coef."""
    vec = tfidf_vectorizer.transform([cleaned_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coefs = logreg_estimator.coef_[0]

    nonzero_cols = vec.nonzero()[1]
    contributions: list[tuple[str, float]] = []
    for col in nonzero_cols:
        score = float(vec[0, col] * coefs[col])
        contributions.append((str(feature_names[col]), score))

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "token": tok,
            "contribution": round(val, 4),
            "direction": "positive" if val >= 0 else "negative",
        }
        for tok, val in contributions[:top_n]
    ]
