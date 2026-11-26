"""Layer 2: Data Layer — Acquire, inspect, deduplicate, and split real movie review data."""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_SUMMARY_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RAW_DATASET_PATH,
    TARGET_SAMPLE_SIZE,
    TEST_RATIO,
    TEST_SPLIT_PATH,
    TRAIN_RATIO,
    TRAIN_SPLIT_PATH,
    VAL_RATIO,
    VAL_SPLIT_PATH,
    ensure_directories,
)

IMDB_TAR_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _download_imdb_sample(target_size: int = TARGET_SAMPLE_SIZE) -> pd.DataFrame:
    """Download Stanford IMDb aclImdb_v1.tar.gz and extract a balanced stratified sample."""
    archive_path = RAW_DATA_DIR / "aclImdb_v1.tar.gz"
    if not archive_path.exists():
        print(f"Downloading Stanford IMDb dataset from {IMDB_TAR_URL} ...")
        req = urllib.request.Request(
            IMDB_TAR_URL,
            headers={"User-Agent": "CinePulse-NLP-Capstone/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(archive_path, "wb") as out:
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    print(f"  Downloaded {downloaded // (1024 * 1024)} MB...", flush=True)

    half = target_size // 2
    per_split_per_class = half // 2  # pull evenly from official train and test folders

    records: list[dict[str, object]] = []
    counts = {("train", "pos"): 0, ("train", "neg"): 0, ("test", "pos"): 0, ("test", "neg"): 0}

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            parts = member.name.split("/")
            if len(parts) < 4:
                continue
            split_name, label_name = parts[1], parts[2]
            key = (split_name, label_name)
            if key not in counts or counts[key] >= per_split_per_class:
                if all(v >= per_split_per_class for v in counts.values()):
                    break
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode("utf-8", errors="replace").strip()
            if len(text) < 25:
                continue

            # Filename format: <id>_<rating>.txt
            stem = Path(parts[-1]).stem
            rating_str = stem.split("_")[-1] if "_" in stem else "0"
            rating = int(rating_str) if rating_str.isdigit() else (8 if label_name == "pos" else 3)

            records.append(
                {
                    "review_id": f"{split_name}_{label_name}_{stem}",
                    "review_text": text,
                    "sentiment": "positive" if label_name == "pos" else "negative",
                    "label": 1 if label_name == "pos" else 0,
                    "star_rating_raw": rating,
                    "source_partition": split_name,
                    "dataset_source": "Stanford IMDb (Maas et al., ACL 2011)",
                }
            )
            counts[key] += 1

    # Remove huge archive after extraction so repo stays clean
    if archive_path.exists():
        archive_path.unlink()

    return pd.DataFrame(records)


def _fallback_nltk_movie_reviews() -> pd.DataFrame:
    """Fallback to NLTK Cornell Movie Reviews (Pang & Lee, 2004) if IMDb download fails."""
    nltk.download("movie_reviews", quiet=True)
    from nltk.corpus import movie_reviews

    records = []
    for category in ("pos", "neg"):
        for idx, fileid in enumerate(movie_reviews.fileids(category)):
            text = movie_reviews.raw(fileid).strip()
            records.append(
                {
                    "review_id": f"cornell_{category}_{idx}",
                    "review_text": text,
                    "sentiment": "positive" if category == "pos" else "negative",
                    "label": 1 if category == "pos" else 0,
                    "star_rating_raw": 8 if category == "pos" else 3,
                    "source_partition": "cornell_v2",
                    "dataset_source": "Cornell Movie Review Polarity v2.0 (Pang & Lee, 2004)",
                }
            )
    return pd.DataFrame(records)


def load_or_fetch_raw_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Load cached raw dataset or acquire real public movie reviews."""
    ensure_directories()
    if RAW_DATASET_PATH.exists() and not force_refresh:
        df = pd.read_csv(RAW_DATASET_PATH)
        if not df.empty:
            return df

    try:
        df = _download_imdb_sample(TARGET_SAMPLE_SIZE)
    except Exception as exc:
        print(f"IMDb download failed ({exc}); falling back to NLTK Cornell Movie Reviews.")
        df = _fallback_nltk_movie_reviews()

    df.to_csv(RAW_DATASET_PATH, index=False)
    return df


def prepare_and_split_dataset(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Inspect, clean missing/duplicate rows, and create stratified train/val/test splits."""
    ensure_directories()
    raw_df = load_or_fetch_raw_dataset(force_refresh=force_refresh)
    initial_rows = len(raw_df)

    # Drop missing or blank review texts
    df = raw_df.dropna(subset=["review_text", "label"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() >= 20]
    after_null_clean = len(df)

    # Deduplicate on normalized lowercase whitespace-collapsed text
    df["_norm_key"] = (
        df["review_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(subset=["_norm_key"]).drop(columns=["_norm_key"]).reset_index(drop=True)
    duplicates_removed = after_null_clean - len(df)

    # Compute descriptive text length features
    df["char_length"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].str.split().str.len()

    # Stratified Split: Train (70%), Validation (15%), Test (15%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    summary = {
        "dataset_name": str(df["dataset_source"].iloc[0]),
        "source_url": "https://ai.stanford.edu/~amaas/data/sentiment/",
        "initial_records": int(initial_rows),
        "cleaned_records": int(len(df)),
        "missing_or_short_removed": int(initial_rows - after_null_clean),
        "duplicates_removed": int(duplicates_removed),
        "positive_count": int((df["label"] == 1).sum()),
        "negative_count": int((df["label"] == 0).sum()),
        "splits": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "word_count_stats": {
            "mean": round(float(df["word_count"].mean()), 1),
            "median": round(float(df["word_count"].median()), 1),
            "min": int(df["word_count"].min()),
            "max": int(df["word_count"].max()),
        },
        "rating_distribution": {
            str(k): int(v)
            for k, v in df["star_rating_raw"].value_counts().sort_index().items()
        },
        "sample_records": [
            {
                "review_id": str(row["review_id"]),
                "sentiment": str(row["sentiment"]),
                "star_rating_raw": int(row["star_rating_raw"]),
                "word_count": int(row["word_count"]),
                "excerpt": str(row["review_text"])[:220].replace("\n", " ") + "...",
            }
            for _, row in df.sample(n=min(6, len(df)), random_state=RANDOM_STATE).iterrows()
        ],
    }

    with open(DATASET_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"full": df, "train": train_df, "val": val_df, "test": test_df}
