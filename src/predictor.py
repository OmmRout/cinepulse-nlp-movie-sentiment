"""Inference & explainability service for single reviews and batch CSV/TXT uploads."""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np

from src.classical_models import explain_single_review_tokens
from src.config import CLASSICAL_BUNDLE_PATH, TRANSFORMER_PRELOAD
from src.preprocessing import ReviewPreprocessor
from src.security import sanitize_display_text
from src.transformer_model import (
    DistilBertSentimentEngine,
    analyze_movie_aspects,
    estimate_star_rating,
)


class SentimentService:
    """Thread-safe service coordinating classical ML and DistilBERT ONNX inference."""

    def __init__(self) -> None:
        self.preprocessor = ReviewPreprocessor()
        self.bundle: dict[str, Any] | None = None
        self.transformer = DistilBertSentimentEngine()
        self._load_classical()
        if TRANSFORMER_PRELOAD:
            self.transformer.load()

    def _load_classical(self) -> None:
        if CLASSICAL_BUNDLE_PATH.exists():
            self.bundle = joblib.load(CLASSICAL_BUNDLE_PATH)

    @property
    def is_ready(self) -> bool:
        return self.bundle is not None

    def _vectorize(self, cleaned_texts: list[str]) -> dict[str, Any]:
        assert self.bundle is not None
        bow = self.bundle["bow_vectorizer"].transform(cleaned_texts)
        tfidf = self.bundle["tfidf_vectorizer"].transform(cleaned_texts)
        dense = self.bundle["embed_normalizer"].transform(
            self.bundle["svd_embedder"].transform(tfidf)
        )
        return {"bow": bow, "tfidf": tfidf, "dense": dense}

    def predict_single(self, raw_text: str, model_key: str = "ensemble") -> dict[str, Any]:
        """Predict sentiment, model breakdown, token attribution, and aspects for one review."""
        if not self.is_ready:
            self._load_classical()
        if not self.is_ready:
            raise RuntimeError("Trained NLP model bundle not found.")

        assert self.bundle is not None
        safe_text = sanitize_display_text(raw_text)
        trace = self.preprocessor.preprocess_with_trace(raw_text)
        feats = self._vectorize([trace.cleaned_text])

        model_scores: list[dict[str, Any]] = []
        prob_map: dict[str, float] = {}

        for key, info in self.bundle["models"].items():
            t0 = time.perf_counter()
            X = feats[info["feature_space"]]
            prob_pos = float(info["estimator"].predict_proba(X)[0, 1])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            prob_map[key] = prob_pos
            model_scores.append(
                {
                    "key": key,
                    "name": info["name"],
                    "family": "Classical ML",
                    "prob_positive": round(prob_pos, 4),
                    "prob_negative": round(1.0 - prob_pos, 4),
                    "prediction": "Positive" if prob_pos >= 0.5 else "Negative",
                    "confidence": round(max(prob_pos, 1.0 - prob_pos), 4),
                    "latency_ms": round(latency_ms, 2),
                }
            )

        # Transformer inference (DistilBERT ONNX)
        transformer_active = False
        try:
            tr_prob, tr_ms = self.transformer.predict_proba_positive(safe_text)
            prob_map["distilbert_onnx"] = tr_prob
            transformer_active = True
            model_scores.append(
                {
                    "key": "distilbert_onnx",
                    "name": "DistilBERT (SST-2 ONNX)",
                    "family": "Transformer",
                    "prob_positive": round(tr_prob, 4),
                    "prob_negative": round(1.0 - tr_prob, 4),
                    "prediction": "Positive" if tr_prob >= 0.5 else "Negative",
                    "confidence": round(max(tr_prob, 1.0 - tr_prob), 4),
                    "latency_ms": round(tr_ms, 2),
                }
            )
        except Exception as exc:
            model_scores.append(
                {
                    "key": "distilbert_onnx",
                    "name": "DistilBERT (SST-2 ONNX)",
                    "family": "Transformer",
                    "unavailable": True,
                    "reason": str(exc),
                }
            )

        # Choose final verdict probability
        if model_key in prob_map:
            final_prob = prob_map[model_key]
            active_model_label = next(
                m["name"] for m in model_scores if m["key"] == model_key
            )
        else:
            # Weighted Ensemble: 60% DistilBERT + 25% TF-IDF LogReg + 15% TF-IDF SVM
            if transformer_active:
                final_prob = (
                    0.60 * prob_map["distilbert_onnx"]
                    + 0.25 * prob_map["tfidf_logreg"]
                    + 0.15 * prob_map["tfidf_svm"]
                )
                active_model_label = "CinePulse Ensemble (DistilBERT + LogReg + SVM)"
            else:
                final_prob = 0.60 * prob_map["tfidf_logreg"] + 0.40 * prob_map["tfidf_svm"]
                active_model_label = "Classical Ensemble (LogReg + SVM)"

        verdict = "Positive" if final_prob >= 0.5 else "Negative"
        confidence = max(final_prob, 1.0 - final_prob)
        if 0.44 <= final_prob <= 0.56:
            intensity = "Mixed / Borderline"
        elif confidence >= 0.85:
            intensity = "Strong"
        elif confidence >= 0.68:
            intensity = "Moderate"
        else:
            intensity = "Mild"

        token_attributions = explain_single_review_tokens(
            trace.cleaned_text,
            self.bundle["tfidf_vectorizer"],
            self.bundle["models"]["tfidf_logreg"]["estimator"],
            top_n=10,
        )
        aspects = analyze_movie_aspects(safe_text)

        return {
            "verdict": verdict,
            "intensity": intensity,
            "active_model": active_model_label,
            "prob_positive": round(final_prob, 4),
            "prob_negative": round(1.0 - final_prob, 4),
            "confidence": round(confidence, 4),
            "estimated_rating": estimate_star_rating(final_prob),
            "word_count": len(safe_text.split()),
            "preprocessing": {
                "normalized_preview": trace.normalized_text[:320],
                "token_count_raw": len(trace.tokens_raw),
                "token_count_after_stopwords": len(trace.tokens_filtered),
                "lemmatized_sample": trace.tokens_lemmatized[:24],
            },
            "model_comparison": model_scores,
            "token_attributions": token_attributions,
            "aspects": aspects,
        }

    def predict_batch(self, raw_texts: list[str], model_key: str = "tfidf_logreg") -> dict[str, Any]:
        """Score a batch of uploaded movie reviews and compute audience opinion distribution."""
        if not self.is_ready:
            self._load_classical()
        assert self.bundle is not None

        sanitized_texts = [sanitize_display_text(t) for t in raw_texts]
        cleaned_texts = self.preprocessor.transform_corpus(sanitized_texts)
        feats = self._vectorize(cleaned_texts)

        used_transformer = False
        if model_key == "distilbert_onnx":
            try:
                probs, _ = self.transformer.predict_batch(sanitized_texts, batch_size=8)
                used_transformer = True
                model_label = "DistilBERT (SST-2 ONNX)"
            except Exception:
                model_key = "tfidf_logreg"

        if not used_transformer:
            if model_key not in self.bundle["models"]:
                model_key = "tfidf_logreg"
            info = self.bundle["models"][model_key]
            probs = info["estimator"].predict_proba(feats[info["feature_space"]])[:, 1]
            model_label = info["name"]

        rows: list[dict[str, Any]] = []
        aspect_totals: dict[str, dict[str, int]] = {}
        rating_bins = {f"{i}-{i+1}": 0 for i in range(1, 10)}

        for idx, (raw, prob) in enumerate(zip(sanitized_texts, probs), start=1):
            p = float(prob)
            verdict = "Positive" if p >= 0.5 else "Negative"
            conf = max(p, 1.0 - p)
            rating = estimate_star_rating(p)
            aspects = analyze_movie_aspects(raw)
            for asp in aspects:
                entry = aspect_totals.setdefault(asp["aspect"], {"mentions": 0, "positive": 0})
                entry["mentions"] += 1
                if asp["tone"] == "Positive":
                    entry["positive"] += 1

            bin_low = min(9, max(1, int(rating)))
            rating_bins[f"{bin_low}-{bin_low+1}"] += 1

            rows.append(
                {
                    "index": idx,
                    "excerpt": raw[:180] + ("..." if len(raw) > 180 else ""),
                    "full_text": raw,
                    "sentiment": verdict,
                    "prob_positive": round(p, 4),
                    "confidence": round(conf, 4),
                    "estimated_rating": rating,
                    "aspects": [a["aspect"] for a in aspects],
                }
            )

        pos_count = sum(1 for r in rows if r["sentiment"] == "Positive")
        neg_count = len(rows) - pos_count
        avg_rating = round(float(np.mean([r["estimated_rating"] for r in rows])), 2)
        avg_conf = round(float(np.mean([r["confidence"] for r in rows])), 4)

        aspect_summary = [
            {
                "aspect": k,
                "mentions": v["mentions"],
                "positive_share": round(v["positive"] / max(v["mentions"], 1), 3),
            }
            for k, v in sorted(aspect_totals.items(), key=lambda x: x[1]["mentions"], reverse=True)
        ]

        return {
            "model_used": model_label,
            "total_reviews": len(rows),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_pct": round(100.0 * pos_count / max(len(rows), 1), 1),
            "avg_estimated_rating": avg_rating,
            "avg_confidence": avg_conf,
            "rating_histogram": rating_bins,
            "aspect_summary": aspect_summary,
            "rows": rows,
        }
