"""End-to-End Capstone NLP Pipeline Runner (Layers 2 through 8)."""

from __future__ import annotations

import joblib

from src.classical_models import extract_top_ngrams, train_classical_models
from src.config import CLASSICAL_BUNDLE_PATH, ensure_directories
from src.data_loader import prepare_and_split_dataset
from src.evaluation import evaluate_all_models
from src.preprocessing import ReviewPreprocessor
from src.representations import build_representations
from src.transformer_model import DistilBertSentimentEngine


def run_full_pipeline(force_refresh_data: bool = False) -> dict:
    """Execute complete NLP pipeline and persist models, metrics, plots, and error analysis."""
    ensure_directories()
    print("[1/5] Preparing and splitting real IMDb movie review dataset...")
    splits = prepare_and_split_dataset(force_refresh=force_refresh_data)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    print("[2/5] Running negation-aware preprocessing and WordNet lemmatization...")
    preprocessor = ReviewPreprocessor()
    train_clean = preprocessor.transform_corpus(train_df["review_text"].tolist())
    val_clean = preprocessor.transform_corpus(val_df["review_text"].tolist())
    test_clean = preprocessor.transform_corpus(test_df["review_text"].tolist())

    print("[3/5] Building BoW, TF-IDF, and 128-dim SVD embedding representations...")
    repr_artifacts, matrices = build_representations(train_clean, val_clean, test_clean)

    print("[4/5] Training Classical ML baselines (NB, Logistic Regression, Linear SVM)...")
    trained_classical = train_classical_models(
        matrices,
        train_df["label"].to_numpy(dtype=int),
        val_df["label"].to_numpy(dtype=int),
    )
    top_ngrams = extract_top_ngrams(
        repr_artifacts.tfidf_vectorizer,
        trained_classical["tfidf_logreg"]["estimator"],
        top_k=18,
    )

    bundle = {
        "bow_vectorizer": repr_artifacts.bow_vectorizer,
        "tfidf_vectorizer": repr_artifacts.tfidf_vectorizer,
        "svd_embedder": repr_artifacts.svd_embedder,
        "embed_normalizer": repr_artifacts.embed_normalizer,
        "models": {
            key: {
                "name": info["name"],
                "representation": info["representation"],
                "feature_space": info["feature_space"],
                "estimator": info["estimator"],
            }
            for key, info in trained_classical.items()
        },
    }
    joblib.dump(bundle, CLASSICAL_BUNDLE_PATH, compress=3)
    print(f"Saved classical bundle to {CLASSICAL_BUNDLE_PATH}")

    print("[5/5] Benchmarking Classical Models + DistilBERT Transformer & Error Analysis...")
    transformer = DistilBertSentimentEngine()
    report = evaluate_all_models(
        test_df=test_df,
        matrices=matrices,
        trained_classical=trained_classical,
        top_ngrams=top_ngrams,
        transformer_engine=transformer,
    )

    print("\n=== Test Split Model Benchmark ===")
    for m in report["models"]:
        met = m["metrics"]
        print(
            f"- {m['name']:30s} | Acc={met['accuracy']:.4f} | "
            f"F1={met['f1_score']:.4f} | AUC={met['roc_auc']:.4f} | "
            f"Latency={met['latency_ms_per_review']:.2f} ms/rev"
        )
    return report


if __name__ == "__main__":
    run_full_pipeline()
