# CinePulse NLP — Movie Review Sentiment Dashboard

**Course:** ETMMTI303 – Language Technologies and Text Intelligence (MCA AI & ML, Semester III)  
**Selected Topic:** Option 12 — Movie Review Sentiment Dashboard  
**Stack:** Python 3.12 · Flask · scikit-learn · NLTK · Quantized DistilBERT (ONNX Runtime) · Chart.js · Gunicorn  

---

## Overview

CinePulse NLP is an end-to-end, security-hardened Natural Language Processing web application that classifies movie reviews, extracts aspect-level audience sentiment (Acting, Plot, Direction & Pacing, Visuals, Soundtrack), compares **5 classical ML pipelines** against a **pretrained DistilBERT Transformer**, performs automated error taxonomy analysis, and visualizes audience opinion patterns inside an interactive **Warm Obsidian & Champagne Gold** cinema dashboard.

## 11-Layer Architecture (ETMMTI303 Standard Blueprint)

1. **Problem & Input Layer** — Classify unseen movie reviews (single text or batch `.csv`/`.txt`) into positive/negative sentiment, calibrated probability, estimated 1–10 rating, and aspect-level tone.
2. **Data Layer (`src/data_loader.py`)** — Acquires 6,000 balanced real reviews from the **Stanford IMDb Large Movie Review Dataset** (`aclImdb v1.0`), removes duplicates/blanks, and builds deterministic stratified splits (`70%` Train / `15%` Validation / `15%` Test).
3. **Preprocessing Layer (`src/preprocessing.py`)** — HTML unescaping & `<br />` stripping, URL cleanup, contraction expansion (`wasn't` → `was not`), **negation-aware stop-word filtering** (preserving `not`, `never`, `but`, `however`, `although`, etc.), and WordNet lemmatization.
4. **Representation Layer (`src/representations.py`)** — Compares three distinct representations:
   - Bag-of-Words (`CountVectorizer`, 1–2 grams)
   - Sublinear TF-IDF (`TfidfVectorizer`, 1–2 grams)
   - Dense Latent Semantic Embeddings (`TruncatedSVD` 128-dim L2-normalized + 2D PCA projection)
5. **Classical Modeling Layer (`src/classical_models.py`)** — Trains BoW + Multinomial NB, TF-IDF + Multinomial NB, TF-IDF + Logistic Regression, Calibrated TF-IDF + Linear SVM, and Dense SVD + Logistic Regression.
6. **Advanced NLP / Transformer Layer (`src/transformer_model.py`)** — Runs CPU-efficient quantized **DistilBERT (`Xenova/distilbert-base-uncased-finetuned-sst-2-english`)** via ONNX Runtime with head+tail chunking for long reviews, alongside a cinematic aspect sentiment analyzer.
7. **Evaluation Layer (`src/evaluation.py`)** — Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, Score Calibration Histograms, and per-review Inference Latency on the exact same unseen test split.
8. **Error Analysis (`src/evaluation.py`)** — Categorizes False Positives, False Negatives, High-Confidence Correct cases, and Transformer-vs-Classical disagreements across linguistic causes (concessive contrast, negation scope, sarcasm, long plot summaries).
9. **Application Layer (`app.py`, `src/security.py`)** — Hardened Flask server with REST APIs (`/api/predict`, `/api/batch`, `/api/overview`, `/api/export`, `/healthz`).
10. **Visualization & Reporting (`templates/index.html`, `static/`)** — Interactive Chart.js benchmarks, ROC curves, 2D PCA embedding clusters, confusion matrix explorer, predictive n-grams, and animated projector/bokeh backdrop.
11. **Final Delivery (`docs/`, `tests/`, `render.yaml`)** — Full academic project report, presentation slide deck, automated `pytest` verification suite, and one-click Render Blueprint deployment.

---

## Security Protocols Implemented

- **Strict Content Security Policy (CSP):** Self-hosted JS & vendored Chart.js (`script-src 'self'`), `frame-ancestors 'none'`, `object-src 'none'`.
- **CSRF & Same-Origin Protection:** Session-bound cryptographic tokens (`secrets.token_urlsafe`) verified with constant-time `hmac.compare_digest` via `X-CSRF-Token` on all mutating routes.
- **Rate Limiting:** Sliding-window limits via `Flask-Limiter` (`60/min` on `/api/predict`, `12/min` on `/api/batch`, `20/min` on `/api/export`).
- **Input & Upload Validation:** 10,000-char text cap, null-byte & control-character rejection, `.csv`/`.txt` whitelist, 2 MB & 500-row upload cap, and `HttpOnly`/`SameSite=Lax`/`Secure` session cookies.
- **XSS & CSV Formula Injection Defense:** Safe DOM rendering via `textContent` (no untrusted `innerHTML`) and OWASP formula-prefix neutralization (`=`, `+`, `-`, `@`) on CSV downloads.
- **HTTP Hardening Headers:** `HSTS`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, and stripped `Server` banner.

---

## Local Setup & Execution

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the end-to-end NLP training & evaluation pipeline
python -m src.pipeline

# 4. Run automated unit, model, and security tests
pytest -v

# 5. Start the dashboard locally
python app.py
# Open http://localhost:5000
```

## Deploying to Render

This repository includes `render.yaml`, `Procfile`, and `.python-version`:
1. Push the repository to GitHub.
2. In Render Dashboard, choose **New + -> Blueprint** and connect the repository.
3. Render automatically installs dependencies, preloads assets via `python -m src.build_assets`, provisions a cryptographic `SECRET_KEY`, and launches Gunicorn with `/healthz` health checks.
