"""Build-time asset initializer for Render deployment and fresh clones."""

from __future__ import annotations

from src.config import CLASSICAL_BUNDLE_PATH, EVALUATION_REPORT_PATH, TRANSFORMER_PRELOAD
from src.pipeline import run_full_pipeline
from src.preprocessing import ensure_nltk_resources
from src.transformer_model import DistilBertSentimentEngine


def main() -> None:
    ensure_nltk_resources()
    if TRANSFORMER_PRELOAD:
        engine = DistilBertSentimentEngine()
        loaded = engine.load()
        print(f"DistilBERT ONNX preloaded: {loaded}")

    if not CLASSICAL_BUNDLE_PATH.exists() or not EVALUATION_REPORT_PATH.exists():
        print("Trained artifacts missing — running full NLP pipeline...")
        run_full_pipeline()
    else:
        print("Verified pre-trained NLP artifacts present.")


if __name__ == "__main__":
    main()
