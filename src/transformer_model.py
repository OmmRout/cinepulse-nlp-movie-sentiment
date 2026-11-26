"""Layer 6: Advanced NLP Layer — Quantized DistilBERT ONNX Transformer & Aspect Analyzer."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import HF_CACHE_DIR, TRANSFORMER_MAX_LENGTH, TRANSFORMER_REPO_ID

logger = logging.getLogger("cinepulse.transformer")

# Movie domain aspect lexicon for fine-grained audience opinion breakdown
MOVIE_ASPECTS: dict[str, set[str]] = {
    "Acting & Cast": {
        "actor", "actress", "acting", "cast", "performance", "performances",
        "character", "characters", "chemistry", "villain", "hero", "lead",
        "supporting", "dialogue", "portrayal", "charisma", "delivery",
    },
    "Plot & Screenplay": {
        "plot", "story", "storyline", "script", "screenplay", "writing",
        "writer", "twist", "ending", "narrative", "cliche", "predictable",
        "suspense", "mystery", "characterization", "subplots", "premise",
    },
    "Direction & Pacing": {
        "director", "directing", "direction", "pacing", "paced", "slow",
        "boring", "dragging", "rushed", "editing", "edited", "tight",
        "flow", "vision", "filmmaker", "masterpiece", "execution",
    },
    "Visuals & Cinematography": {
        "cinematography", "visual", "visuals", "effects", "cgi", "vfx",
        "camera", "shot", "shots", "lighting", "color", "aesthetic",
        "scenery", "costume", "production", "atmosphere", "action", "stunts",
    },
    "Soundtrack & Audio": {
        "music", "score", "soundtrack", "song", "songs", "audio",
        "sound", "theme", "composition", "composer", "acoustic", "loud",
    },
}

_POSITIVE_CUES = {
    "brilliant", "great", "amazing", "wonderful", "superb", "excellent",
    "outstanding", "stunning", "masterpiece", "captivating", "compelling",
    "strong", "memorable", "engaging", "beautiful", "fantastic", "solid",
    "touching", "immersive", "sharp", "clever", "enjoyable", "phenomenal",
}

_NEGATIVE_CUES = {
    "terrible", "awful", "boring", "weak", "bad", "poor", "dull",
    "predictable", "mess", "flat", "wooden", "cliche", "tedious",
    "disappointing", "forgettable", "cheap", "unconvincing", "horrible",
    "slow", "annoying", "waste", "worst", "lifeless", "mediocre",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z]+")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class DistilBertSentimentEngine:
    """CPU-efficient DistilBERT SST-2 inference using ONNX Runtime + Rust Tokenizers."""

    def __init__(
        self,
        repo_id: str = TRANSFORMER_REPO_ID,
        cache_dir: Path = HF_CACHE_DIR,
        max_length: int = TRANSFORMER_MAX_LENGTH,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: list[str] = []
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        """Download (if needed) and initialize the quantized DistilBERT ONNX session."""
        if self.is_loaded:
            return True

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="onnx/model_quantized.onnx",
                cache_dir=str(self.cache_dir),
            )
            tokenizer_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="tokenizer.json",
                cache_dir=str(self.cache_dir),
            )

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]

            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=self.max_length)
            tokenizer.enable_padding(length=self.max_length)
            self._tokenizer = tokenizer
            self._load_error = None
            logger.info("Loaded DistilBERT ONNX model from %s", self.repo_id)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Failed to load DistilBERT ONNX model: %s", exc)
            return False

    def _prepare_chunks(self, text: str) -> list[str]:
        """For long reviews, evaluate opening and closing segments to avoid losing conclusion sentiment."""
        words = text.split()
        if len(words) <= 190:
            return [text]
        head = " ".join(words[:160])
        tail = " ".join(words[-160:])
        return [head, tail]

    def predict_proba_positive(self, text: str) -> tuple[float, float]:
        """Return (positive_probability, latency_ms) for a single review."""
        if not self.load():
            raise RuntimeError(f"Transformer model unavailable: {self._load_error}")

        t0 = time.perf_counter()
        chunks = self._prepare_chunks(text)
        encodings = self._tokenizer.encode_batch(chunks)

        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        ort_inputs: dict[str, np.ndarray] = {"input_ids": input_ids}
        if "attention_mask" in self._input_names:
            ort_inputs["attention_mask"] = attention_mask

        logits = self._session.run(None, ort_inputs)[0]
        probs = _softmax(logits)  # [chunk_count, 2] -> index 0: NEG, index 1: POS
        pos_prob = float(np.mean(probs[:, 1]))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return pos_prob, latency_ms

    def predict_batch(self, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, float]:
        """Predict positive probabilities for a list of reviews using batched ONNX inference."""
        if not self.load():
            raise RuntimeError(f"Transformer model unavailable: {self._load_error}")

        t0 = time.perf_counter()
        # Use single-pass 256-token representation for fast batch benchmarking
        all_pos_probs: list[float] = []
        for start in range(0, len(texts), batch_size):
            slice_texts = texts[start : start + batch_size]
            encodings = self._tokenizer.encode_batch(slice_texts)
            input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

            ort_inputs: dict[str, np.ndarray] = {"input_ids": input_ids}
            if "attention_mask" in self._input_names:
                ort_inputs["attention_mask"] = attention_mask

            logits = self._session.run(None, ort_inputs)[0]
            probs = _softmax(logits)
            all_pos_probs.extend(probs[:, 1].tolist())

        total_ms = (time.perf_counter() - t0) * 1000.0
        return np.array(all_pos_probs, dtype=np.float64), total_ms


def analyze_movie_aspects(text: str) -> list[dict[str, Any]]:
    """Extract movie aspect mentions (acting, plot, pacing, visuals, audio) and local polarity."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip()) or [text]
    aspect_results: list[dict[str, Any]] = []

    for aspect_name, keywords in MOVIE_ASPECTS.items():
        mentions = 0
        pos_hits = 0
        neg_hits = 0
        matched_terms: list[str] = []

        for sent in sentences:
            tokens = _WORD_RE.findall(sent.lower())
            token_set = set(tokens)
            overlap = token_set & keywords
            if overlap:
                mentions += 1
                matched_terms.extend(sorted(overlap))
                has_negation = any(w in token_set for w in ("not", "no", "never", "hardly", "barely", "neither"))
                s_pos = len(token_set & _POSITIVE_CUES)
                s_neg = len(token_set & _NEGATIVE_CUES)
                if has_negation:
                    s_pos, s_neg = s_neg, s_pos
                pos_hits += s_pos
                neg_hits += s_neg

        if mentions > 0:
            score = 0.5
            if pos_hits + neg_hits > 0:
                score = (pos_hits + 0.5) / (pos_hits + neg_hits + 1.0)
            tone = (
                "Positive" if score >= 0.6 else ("Negative" if score <= 0.4 else "Neutral / Mixed")
            )
            aspect_results.append(
                {
                    "aspect": aspect_name,
                    "mentions": mentions,
                    "tone": tone,
                    "score": round(score, 3),
                    "matched_keywords": sorted(set(matched_terms))[:5],
                }
            )

    return aspect_results


def estimate_star_rating(pos_prob: float) -> float:
    """Map calibrated positive sentiment probability [0,1] to a 1.0–10.0 IMDb-style rating."""
    rating = 1.0 + 9.0 * float(np.clip(pos_prob, 0.0, 1.0))
    return round(rating, 1)
