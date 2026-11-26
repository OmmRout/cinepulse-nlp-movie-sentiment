"""Central configuration for the Movie Review Sentiment Dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
HF_CACHE_DIR = ARTIFACTS_DIR / "hf_cache"
ONNX_MODEL_DIR = ARTIFACTS_DIR / "onnx_distilbert"

RAW_DATASET_PATH = RAW_DATA_DIR / "movie_reviews_imdb.csv"
TRAIN_SPLIT_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_SPLIT_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_SPLIT_PATH = PROCESSED_DATA_DIR / "test.csv"

CLASSICAL_BUNDLE_PATH = ARTIFACTS_DIR / "classical_models.joblib"
WORDNET_LEXICON_PATH = ARTIFACTS_DIR / "wordnet_noun_lexicon.joblib"
EMBEDDING_PROJECTION_PATH = ARTIFACTS_DIR / "embedding_projection.json"
EVALUATION_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
ERROR_ANALYSIS_PATH = ARTIFACTS_DIR / "error_analysis.json"
DATASET_SUMMARY_PATH = ARTIFACTS_DIR / "dataset_summary.json"
ASPECT_KEYWORDS_PATH = ARTIFACTS_DIR / "aspect_summary.json"

# Reproducibility & Split Configuration
RANDOM_STATE = 42
TARGET_SAMPLE_SIZE = 6000  # Balanced 3,000 positive + 3,000 negative real IMDb reviews
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRANSFORMER_EVAL_LIMIT = 600  # Stratified test subset for transformer benchmarking

# Transformer Configuration (Quantized ONNX DistilBERT SST-2 for low-RAM Render hosting)
TRANSFORMER_REPO_ID = os.getenv(
    "TRANSFORMER_REPO_ID",
    "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
)
TRANSFORMER_PRELOAD = os.getenv("TRANSFORMER_PRELOAD", "true").lower() == "true"
TRANSFORMER_MAX_LENGTH = 256

# Security Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development").lower()
IS_PRODUCTION = FLASK_ENV == "production"
MAX_REVIEW_CHARS = int(os.getenv("MAX_REVIEW_CHARS", "10000"))
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "500"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".txt"}
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")


def ensure_directories() -> None:
    """Create required project directories if missing."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        PLOTS_DIR,
        HF_CACHE_DIR,
        ONNX_MODEL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
