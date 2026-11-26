"""Integration tests for trained NLP models, representations, and aspect extraction."""

from src.config import CLASSICAL_BUNDLE_PATH, EVALUATION_REPORT_PATH
from src.predictor import SentimentService
from src.transformer_model import analyze_movie_aspects, estimate_star_rating


def test_artifacts_exist_and_predict():
    assert CLASSICAL_BUNDLE_PATH.exists(), "Classical model bundle must be generated first."
    assert EVALUATION_REPORT_PATH.exists(), "Evaluation report JSON must exist."

    service = SentimentService()
    pos_res = service.predict_single(
        "Brilliant acting, gorgeous cinematography, and an unforgettable masterpiece of a screenplay."
    )
    neg_res = service.predict_single(
        "Terrible acting, awful boring plot, and the worst dialogue I have seen all year."
    )

    assert pos_res["verdict"] == "Positive"
    assert pos_res["prob_positive"] > 0.65
    assert neg_res["verdict"] == "Negative"
    assert neg_res["prob_positive"] < 0.35
    assert len(pos_res["token_attributions"]) > 0
    assert len(pos_res["model_comparison"]) >= 5


def test_aspect_extraction_and_rating_mapping():
    aspects = analyze_movie_aspects(
        "The acting was phenomenal, though the pacing felt slow and the soundtrack was forgettable."
    )
    names = {a["aspect"] for a in aspects}
    assert "Acting & Cast" in names
    assert "Direction & Pacing" in names
    assert "Soundtrack & Audio" in names
    assert estimate_star_rating(0.0) == 1.0
    assert estimate_star_rating(1.0) == 10.0
