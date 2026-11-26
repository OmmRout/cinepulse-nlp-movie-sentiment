"""Application security protocols: headers, CSRF, input validation, and sanitization."""

from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from flask import Flask, Request, Response, abort, jsonify, request, session
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.config import (
    ALLOWED_ORIGINS,
    ALLOWED_UPLOAD_EXTENSIONS,
    IS_PRODUCTION,
    MAX_BATCH_ROWS,
    MAX_REVIEW_CHARS,
    MAX_UPLOAD_BYTES,
)

security_logger = logging.getLogger("cinepulse.security")

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def configure_app_security(app: Flask) -> None:
    """Configure hardened Flask cookies, payload limits, and HTTP security headers."""
    secret = os.getenv("SECRET_KEY", "").strip()
    if not secret:
        if IS_PRODUCTION:
            # Generate ephemeral strong key if platform env var wasn't injected yet
            secret = secrets.token_hex(32)
            security_logger.warning(
                "SECRET_KEY missing in production env; generated ephemeral runtime key."
            )
        else:
            secret = secrets.token_hex(32)

    app.config["SECRET_KEY"] = secret
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = IS_PRODUCTION
    app.config["JSON_SORT_KEYS"] = False

    @app.after_request
    def apply_security_headers(response: Response) -> Response:
        # Strict Content Security Policy (self-hosted scripts; Google Fonts for typography)
        csp = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'"
        )
        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        if IS_PRODUCTION or request.is_secure:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"

        # Hide server implementation details
        response.headers.pop("Server", None)
        return response

    @app.errorhandler(413)
    def handle_payload_too_large(_err) -> tuple[Response, int]:
        return (
            jsonify(
                {
                    "error": "Payload exceeds maximum allowed upload limit.",
                    "max_bytes": MAX_UPLOAD_BYTES,
                }
            ),
            413,
        )

    @app.errorhandler(429)
    def handle_rate_limit(_err) -> tuple[Response, int]:
        security_logger.warning(
            "Rate limit exceeded for path=%s remote=%s",
            request.path,
            get_client_ip(request),
        )
        return (
            jsonify(
                {
                    "error": "Rate limit exceeded. Please wait briefly before retrying.",
                }
            ),
            429,
        )


def get_client_ip(req: Request) -> str:
    """Resolve client IP safely behind Render/reverse proxy without trusting spoofed chains blindly."""
    forwarded = req.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return req.remote_addr or "127.0.0.1"


def get_or_create_csrf_token() -> str:
    """Return the session-bound CSRF token, creating one if absent."""
    token = session.get("_csrf_token")
    if not token or not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _is_same_origin(req: Request) -> bool:
    """Verify Origin/Referer matches the request host or configured allowlist."""
    origin = req.headers.get("Origin")
    if not origin:
        return True
    if origin in ALLOWED_ORIGINS:
        return True
    parsed_origin = urlparse(origin)
    request_host = req.host.split(":")[0]
    origin_host = (parsed_origin.hostname or "").lower()
    return origin_host == request_host.lower()


def require_csrf(fn: Callable) -> Callable:
    """Decorator enforcing CSRF token validation and same-origin checks on mutating API calls."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _is_same_origin(request):
            security_logger.warning("Blocked cross-origin request to %s", request.path)
            return jsonify({"error": "Cross-origin request denied."}), 403

        expected = session.get("_csrf_token", "")
        provided = (
            request.headers.get("X-CSRF-Token")
            or request.form.get("csrf_token")
            or ""
        )
        if not expected or not provided or not hmac.compare_digest(expected, provided):
            security_logger.warning("Invalid or missing CSRF token on %s", request.path)
            return jsonify({"error": "Invalid or missing CSRF security token."}), 403

        return fn(*args, **kwargs)

    return wrapper


def validate_review_text(raw_text: object) -> str:
    """Validate and sanitize a single user-submitted movie review string."""
    if not isinstance(raw_text, str):
        raise ValueError("Review text must be a valid UTF-8 string.")

    if "\x00" in raw_text:
        raise ValueError("Null bytes are not permitted in input text.")

    cleaned = _CONTROL_CHAR_RE.sub("", raw_text).strip()
    if not cleaned:
        raise ValueError("Please enter a non-empty movie review.")

    if len(cleaned) > MAX_REVIEW_CHARS:
        raise ValueError(
            f"Review exceeds maximum length of {MAX_REVIEW_CHARS:,} characters."
        )

    return cleaned


def sanitize_display_text(text: str) -> str:
    """Strip HTML tags and control characters for safe plain-text rendering."""
    no_ctrl = _CONTROL_CHAR_RE.sub("", text)
    no_html = _HTML_TAG_RE.sub(" ", no_ctrl)
    return re.sub(r"\s+", " ", no_html).strip()


def sanitize_csv_cell(value: object) -> str:
    """Neutralize spreadsheet formula injection (OWASP CSV Injection defense)."""
    text = str(value if value is not None else "")
    text = _CONTROL_CHAR_RE.sub("", text)
    if text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def validate_uploaded_file(file_storage: FileStorage | None) -> tuple[str, bytes]:
    """Validate uploaded CSV/TXT file extension, size, and encoding safety."""
    if file_storage is None or not file_storage.filename:
        raise ValueError("No file was uploaded.")

    safe_name = secure_filename(file_storage.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("Unsupported file type. Only .csv and .txt files are allowed.")

    raw_bytes = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    if b"\x00" in raw_bytes:
        raise ValueError("Binary or null-byte content detected in uploaded file.")

    return ext, raw_bytes


def enforce_batch_row_limit(rows: Iterable[str]) -> list[str]:
    """Ensure uploaded batch reviews stay bounded within safe processing limits."""
    items = list(rows)
    if not items:
        raise ValueError("No valid review rows were found in the uploaded file.")
    if len(items) > MAX_BATCH_ROWS:
        raise ValueError(
            f"Batch exceeds the maximum limit of {MAX_BATCH_ROWS} reviews per upload."
        )
    return items
