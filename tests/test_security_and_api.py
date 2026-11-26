"""Security protocol and Flask API contract tests."""

import io

import pytest

from app import app
from src.security import sanitize_csv_cell, validate_review_text


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _get_csrf_token(client) -> str:
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    marker = 'name="csrf-token" content="'
    start = html.index(marker) + len(marker)
    end = html.index('"', start)
    return html[start:end]


def test_security_headers_present(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "default-src 'self'" in resp.headers.get("Content-Security-Policy", "")
    assert "Server" not in resp.headers


def test_csrf_enforced_on_predict(client):
    resp = client.post("/api/predict", json={"review_text": "Great movie!"})
    assert resp.status_code == 403


def test_predict_with_valid_csrf(client):
    token = _get_csrf_token(client)
    resp = client.post(
        "/api/predict",
        headers={"X-CSRF-Token": token},
        json={"review_text": "Stunning visuals and heartfelt performances throughout."},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["verdict"] in {"Positive", "Negative"}


def test_input_validation_blocks_null_bytes_and_oversized():
    with pytest.raises(ValueError):
        validate_review_text("hello\x00world")
    with pytest.raises(ValueError):
        validate_review_text("   ")
    with pytest.raises(ValueError):
        validate_review_text("a" * 10005)


def test_csv_formula_injection_sanitized():
    assert sanitize_csv_cell("=cmd|'/C calc'!A0").startswith("'")
    assert sanitize_csv_cell("+1+1").startswith("'")
    assert sanitize_csv_cell("-2+3").startswith("'")
    assert sanitize_csv_cell("@SUM(1,1)").startswith("'")
    assert sanitize_csv_cell("Great film") == "Great film"


def test_batch_upload_rejects_bad_extension(client):
    token = _get_csrf_token(client)
    resp = client.post(
        "/api/batch",
        headers={"X-CSRF-Token": token},
        data={"file": (io.BytesIO(b"hello"), "evil.exe")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_overview_healthz_batch_and_export(client):
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    ov_json = overview.get_json()
    assert len(ov_json["evaluation"]["models"]) >= 6

    token = _get_csrf_token(client)
    csv_payload = b"review\n\"=cmd|calc\"\n\"A wonderful and brilliant cinematic experience!\"\n"
    batch_resp = client.post(
        "/api/batch",
        headers={"X-CSRF-Token": token},
        data={"file": (io.BytesIO(csv_payload), "reviews.csv"), "model": "tfidf_logreg"},
        content_type="multipart/form-data",
    )
    assert batch_resp.status_code == 200
    batch_json = batch_resp.get_json()
    assert batch_json["total_reviews"] == 2

    export_resp = client.post(
        "/api/export",
        headers={"X-CSRF-Token": token},
        json={"rows": batch_json["rows"]},
    )
    assert export_resp.status_code == 200
    exported_csv = export_resp.get_data(as_text=True)
    assert "'=cmd|calc" in exported_csv
