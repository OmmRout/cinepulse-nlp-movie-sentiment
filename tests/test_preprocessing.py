"""Unit tests for Layer 3 preprocessing & negation preservation."""

from src.preprocessing import ReviewPreprocessor


def test_contraction_and_negation_preservation():
    prep = ReviewPreprocessor()
    trace = prep.preprocess_with_trace(
        "<p>I didn't hate the film, and it wasn't boring at all! https://example.com</p>"
    )
    assert "http" not in trace.cleaned_text
    assert "<p>" not in trace.cleaned_text
    assert "not" in trace.tokens_lemmatized
    assert "hate" in trace.tokens_lemmatized
    assert "boring" in trace.tokens_lemmatized


def test_empty_or_punctuation_only():
    prep = ReviewPreprocessor()
    assert prep.transform_one("!!! <br /> ???") == ""
