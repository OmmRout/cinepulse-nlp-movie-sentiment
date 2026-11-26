"""Secured Flask Web Application for CinePulse NLP — Movie Review Sentiment Dashboard."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_limiter import Limiter

from src.config import (
    ASPECT_KEYWORDS_PATH,
    BASE_DIR,
    DATASET_SUMMARY_PATH,
    EMBEDDING_PROJECTION_PATH,
    ERROR_ANALYSIS_PATH,
    EVALUATION_REPORT_PATH,
    IS_PRODUCTION,
    PLOTS_DIR,
    RATELIMIT_STORAGE_URI,
)
from src.predictor import SentimentService
from src.security import (
    configure_app_security,
    enforce_batch_row_limit,
    get_client_ip,
    get_or_create_csrf_token,
    require_csrf,
    sanitize_csv_cell,
    validate_review_text,
    validate_uploaded_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cinepulse.app")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
configure_app_security(app)

limiter = Limiter(
    key_func=lambda: get_client_ip(request),
    app=app,
    default_limits=["200 per minute"],
    storage_uri=RATELIMIT_STORAGE_URI,
)

sentiment_service = SentimentService()


def _load_json_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_reviews_from_upload(ext: str, raw_bytes: bytes) -> list[str]:
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("latin-1")

    reviews: list[str] = []
    if ext == ".txt":
        for line in decoded.splitlines():
            line = line.strip()
            if line:
                reviews.append(validate_review_text(line))
    else:
        reader = csv.DictReader(io.StringIO(decoded))
        if reader.fieldnames:
            candidate_cols = [
                c
                for c in reader.fieldnames
                if c and c.strip().lower() in {"review", "review_text", "text", "comment", "content"}
            ]
            target_col = candidate_cols[0] if candidate_cols else reader.fieldnames[0]
            for row in reader:
                val = (row.get(target_col) or "").strip()
                if val:
                    reviews.append(validate_review_text(val))
        else:
            raw_reader = csv.reader(io.StringIO(decoded))
            for row in raw_reader:
                if row and row[0].strip():
                    reviews.append(validate_review_text(row[0].strip()))

    return enforce_batch_row_limit(reviews)


@app.route("/", methods=["GET"])
def index():
    csrf_token = get_or_create_csrf_token()
    return render_template("index.html", csrf_token=csrf_token)


@app.route("/healthz", methods=["GET"])
@limiter.exempt
def healthz():
    return (
        jsonify(
            {
                "status": "ok" if sentiment_service.is_ready else "degraded",
                "classical_models_ready": sentiment_service.is_ready,
                "transformer_loaded": sentiment_service.transformer.is_loaded,
            }
        ),
        200 if sentiment_service.is_ready else 503,
    )


@app.route("/api/overview", methods=["GET"])
def api_overview():
    return jsonify(
        {
            "dataset": _load_json_artifact(DATASET_SUMMARY_PATH),
            "evaluation": _load_json_artifact(EVALUATION_REPORT_PATH),
            "error_analysis": _load_json_artifact(ERROR_ANALYSIS_PATH),
            "embedding_projection": _load_json_artifact(EMBEDDING_PROJECTION_PATH),
            "aspects": _load_json_artifact(ASPECT_KEYWORDS_PATH),
        }
    )


@app.route("/api/predict", methods=["POST"])
@limiter.limit("60 per minute")
@require_csrf
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        review_text = validate_review_text(payload.get("review_text"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model_choice = str(payload.get("model", "ensemble")).strip()
    try:
        result = sentiment_service.predict_single(review_text, model_key=model_choice)
    except RuntimeError as exc:
        logger.error("Prediction failure: %s", exc)
        return jsonify({"error": "Model service is currently unavailable."}), 503

    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
@limiter.limit("12 per minute")
@require_csrf
def api_batch():
    uploaded = request.files.get("file")
    model_choice = str(request.form.get("model", "tfidf_logreg")).strip()
    try:
        ext, raw_bytes = validate_uploaded_file(uploaded)
        reviews = _extract_reviews_from_upload(ext, raw_bytes)
        result = sentiment_service.predict_batch(reviews, model_key=model_choice)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        logger.error("Batch prediction error: %s", exc)
        return jsonify({"error": "Model service is currently unavailable."}), 503

    return jsonify(result)


@app.route("/api/export", methods=["POST"])
@limiter.limit("20 per minute")
@require_csrf
def api_export():
    """Export batch analysis results as OWASP-sanitized CSV (formula-injection safe)."""
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "No batch rows available for export."}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["Index", "Sentiment", "Positive_Probability", "Confidence", "Estimated_Rating", "Aspects", "Review_Text"]
    )
    for row in rows[:500]:
        if not isinstance(row, dict):
            continue
        writer.writerow(
            [
                sanitize_csv_cell(row.get("index", "")),
                sanitize_csv_cell(row.get("sentiment", "")),
                sanitize_csv_cell(row.get("prob_positive", "")),
                sanitize_csv_cell(row.get("confidence", "")),
                sanitize_csv_cell(row.get("estimated_rating", "")),
                sanitize_csv_cell("; ".join(row.get("aspects", []) or [])),
                sanitize_csv_cell(row.get("full_text", row.get("excerpt", ""))),
            ]
        )

    csv_bytes = output.getvalue().encode("utf-8-sig")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=cinepulse_sentiment_results.csv",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.route("/artifacts/plots/<path:filename>", methods=["GET"])
def serve_plot(filename: str):
    """Serve pre-rendered publication plots safely."""
    allowed_plots = {"model_comparison.png", "roc_curves.png", "confusion_matrices.png"}
    if filename not in allowed_plots:
        return jsonify({"error": "Plot not found."}), 404
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=not IS_PRODUCTION)
