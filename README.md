# CinePulse NLP — Movie Review Sentiment Dashboard

End-to-end NLP sentiment analysis and audience intelligence dashboard built for **ETMMTI303 (Language Technologies and Text Intelligence)** — Topic 12.

- **Live App:** https://cinepulse-nlp-sentiment-dashboard.onrender.com
- **Repository:** https://github.com/OmmRout/cinepulse-nlp-movie-sentiment
- **Stack:** Python 3.12 · Flask · scikit-learn · NLTK · Quantized DistilBERT (ONNX Runtime) · Chart.js · Gunicorn

---

## Features

- **Live Review Analyzer:** Real-time sentiment verdict, calibrated confidence, estimated 1–10 rating, token-level feature attribution, movie aspect breakdown (*Acting, Plot, Direction & Pacing, Visuals, Soundtrack*), and preprocessing audit trace.
- **Batch Screening Studio:** Upload `.csv` or `.txt` files (up to 500 reviews / 2 MB) for polarity distribution, rating histograms, and OWASP-sanitized CSV export.
- **Model Benchmarks & Error Analysis:** Side-by-side comparison of 5 classical ML pipelines and quantized DistilBERT ONNX on `899` unseen Stanford IMDb test reviews, including ROC curves, confusion matrices, 2D PCA embeddings, and failure taxonomy.
- **Cinema UI:** Warm Obsidian, Champagne Gold, and Velvet Crimson interface with an animated `<canvas>` projector-beam and bokeh backdrop.

---

## Test Benchmark (`n = 899` Held-Out IMDb Reviews)

| Model | Representation | Accuracy | F1-Score | ROC-AUC | Latency (ms/rev) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Linear SVM** | Sublinear TF-IDF (1–2g) | **90.21%** | **0.9005** | **0.9611** | 0.003 |
| **TF-IDF + Logistic Regression** | Sublinear TF-IDF (1–2g) | 89.99% | 0.8989 | 0.9591 | 0.001 |
| **TF-IDF + Multinomial NB** | Sublinear TF-IDF (1–2g) | 88.54% | 0.8817 | 0.9520 | 0.001 |
| **DistilBERT (SST-2 ONNX)** | Contextual Subword (768d) | 87.76% | 0.8750 | 0.9472 | 115.53 |
| **BoW + Multinomial NB** | Bag-of-Words (1–2g) | 87.54% | 0.8730 | 0.9369 | 0.002 |
| **Dense SVD + LogReg** | LSA Dense (64-dim) | 87.43% | 0.8703 | 0.9474 | <0.001 |

---

## Pipeline Architecture

| Layer | Module | Summary |
| :--- | :--- | :--- |
| **Data** | `src/data_loader.py` | Loads 6,000 balanced Stanford IMDb reviews (`5,988` after deduplication) into 70/15/15 stratified splits. |
| **Preprocessing** | `src/preprocessing.py` | HTML/URL cleanup, contraction expansion, negation-aware stopwords, and compact WordNet noun lemmatization. |
| **Representations** | `src/representations.py` | Bag-of-Words (12k), Sublinear TF-IDF (15k `float32`), and 64-dim `TruncatedSVD` embeddings + 2D PCA. |
| **Classical Models** | `src/classical_models.py` | Multinomial NB, Logistic Regression, and Calibrated Linear SVM with per-token attribution. |
| **Transformer** | `src/transformer_model.py` | Quantized `distilbert-base-uncased-finetuned-sst-2-english` via ONNX Runtime + movie aspect analyzer. |
| **Evaluation** | `src/evaluation.py` | Metrics, ROC curves, confusion matrices, plots, and automated linguistic error taxonomy. |
| **Web & Security** | `app.py`, `src/security.py` | Hardened Flask API, CSRF verification, rate limiting, CSP, input validation, and CSV injection defense. |

---

## Security Controls

- **Content Security Policy:** Self-hosted scripts and vendored Chart.js (`script-src 'self'`), `frame-ancestors 'none'`, `object-src 'none'`.
- **CSRF & Origin Validation:** Session-bound cryptographic tokens verified via `hmac.compare_digest` (`X-CSRF-Token`) on all `POST` endpoints.
- **Rate Limiting & Input Caps:** `Flask-Limiter` (`60/min` predict, `12/min` batch, `20/min` export), 10,000-character limit, `.csv`/`.txt` allowlist, and 2 MB / 500-row upload cap.
- **Injection & Header Hardening:** Safe DOM `.textContent` rendering, OWASP CSV formula escaping (`=`, `+`, `-`, `@`), `HSTS`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `HttpOnly`/`SameSite=Lax` cookies.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run the full training & evaluation pipeline
python -m src.pipeline

# 4. Run the test suite
pytest -v

# 5. Start the local server (http://localhost:5000)
python app.py
```

---

## Project Structure

```text
├── app.py                  # Flask application & REST endpoints
├── render.yaml             # Render Blueprint deployment config
├── Procfile                # Gunicorn startup command
├── requirements.txt        # Python dependencies
├── data/                   # Dataset documentation, raw sample & 70/15/15 splits
├── src/                    # Modular 11-layer NLP pipeline & security middleware
├── artifacts/              # Trained models, metrics JSON, error analysis & plots
├── static/                 # Warm Obsidian CSS, canvas animation & dashboard JS
├── templates/              # Jinja2 HTML template
├── docs/                   # Academic Capstone Report & Presentation outline
└── tests/                  # Pytest unit, model & security tests
```

## Documentation

- [Project Report](docs/PROJECT_REPORT.md)
- [Presentation Outline](docs/PRESENTATION.md)
- [Dataset Dictionary](data/README_DATA.md)
