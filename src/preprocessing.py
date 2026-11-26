"""Layer 3: Preprocessing Layer — Normalization, negation-aware stopwords, and lemmatization."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preserve crucial negation and contrast markers that flip sentiment polarity
NEGATION_AND_CONTRAST_KEEP = {
    "no",
    "nor",
    "not",
    "never",
    "neither",
    "none",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "scarcely",
    "but",
    "however",
    "although",
    "though",
    "yet",
    "against",
    "without",
    "very",
    "too",
    "most",
    "more",
    "less",
}

CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "shan't": "shall not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}

_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def ensure_nltk_resources() -> None:
    """Download required lightweight NLTK corpora if not already cached."""
    for pkg, path in (
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@dataclass
class PreprocessingTrace:
    """Structured audit trace showing every preprocessing transformation on an input text."""

    raw_text: str
    normalized_text: str
    tokens_raw: list[str]
    tokens_filtered: list[str]
    tokens_lemmatized: list[str]
    cleaned_text: str


class ReviewPreprocessor:
    """Negation-preserving NLP text preprocessor for movie reviews."""

    def __init__(self) -> None:
        ensure_nltk_resources()
        base_stops = set(stopwords.words("english"))
        self.stop_words = base_stops - NEGATION_AND_CONTRAST_KEEP
        self.lemmatizer = WordNetLemmatizer()

    def normalize(self, text: str) -> str:
        """Unescape HTML, strip tags/URLs, expand contractions, and lowercase."""
        cleaned = html.unescape(text or "")
        cleaned = _HTML_BREAK_RE.sub(" ", cleaned)
        cleaned = _HTML_TAG_RE.sub(" ", cleaned)
        cleaned = _URL_RE.sub(" ", cleaned)
        cleaned = cleaned.lower()

        for pattern, replacement in CONTRACTIONS.items():
            cleaned = cleaned.replace(pattern, replacement)

        cleaned = _NON_ALPHA_RE.sub(" ", cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    def preprocess_with_trace(self, text: str) -> PreprocessingTrace:
        """Run full preprocessing pipeline and return intermediate representations."""
        normalized = self.normalize(text)
        tokens_raw = normalized.split()
        tokens_filtered = [
            tok for tok in tokens_raw if tok not in self.stop_words and len(tok) > 1
        ]
        tokens_lemmatized = [self.lemmatizer.lemmatize(tok) for tok in tokens_filtered]
        cleaned_text = " ".join(tokens_lemmatized)
        return PreprocessingTrace(
            raw_text=text,
            normalized_text=normalized,
            tokens_raw=tokens_raw,
            tokens_filtered=tokens_filtered,
            tokens_lemmatized=tokens_lemmatized,
            cleaned_text=cleaned_text,
        )

    def transform_one(self, text: str) -> str:
        """Return cleaned lemmatized text for a single review."""
        return self.preprocess_with_trace(text).cleaned_text

    def transform_corpus(self, texts: list[str] | tuple[str, ...]) -> list[str]:
        """Transform a list of reviews efficiently."""
        return [self.transform_one(t) for t in texts]
