# CinePulse NLP — Capstone Presentation Outline

**Course:** ETMMTI303 – Language Technologies and Text Intelligence  
**Topic 12:** Movie Review Sentiment Dashboard  

---

## Slide 1: Title & Problem Statement
- **Title:** CinePulse NLP — End-to-End Movie Review Sentiment Intelligence Dashboard
- **Use Case:** Studios, streaming platforms, and film critics need to convert unstructured audience reviews into polarity verdicts, estimated rating distributions, and aspect-level feedback (acting, plot, pacing, visuals, audio).
- **Expected Input / Output:** Raw review text or `.csv`/`.txt` batches → Sentiment verdict, confidence score, estimated 1–10 rating, token attributions, aspect breakdown, and comparative benchmarks.

## Slide 2: Dataset & Splitting Strategy
- **Source:** Stanford IMDb Large Movie Review Dataset (`aclImdb v1.0`, Maas et al., ACL 2011).
- **Curation:** Balanced sample of 6,000 real reviews (3,000 positive / 3,000 negative), deduplicated on normalized text, split strictly into **70% Train / 15% Validation / 15% Test** with zero data leakage.

## Slide 3: Preprocessing & Text Representations
- **Preprocessing:** HTML tag/entity cleanup, contraction expansion (`wasn't` → `was not`), **negation-aware stop-word removal** (retaining `not`, `never`, `but`, `however`), and WordNet lemmatization.
- **3 Representations Compared:**
  1. Bag-of-Words (1–2 grams, 20k features)
  2. Sublinear TF-IDF (1–2 grams, 25k features)
  3. Dense LSA Embeddings (`TruncatedSVD` 128-dim L2-normalized + 2D PCA cluster plot)
  4. Contextual Subword Embeddings (DistilBERT 768-dim)

## Slide 4: Classical Baselines vs. DistilBERT Transformer
- Trained **5 Classical Pipelines** (`BoW + NB`, `TF-IDF + NB`, `TF-IDF + LogReg`, `TF-IDF + Linear SVM`, `SVD + LogReg`) alongside **Quantized DistilBERT SST-2 (ONNX Runtime)**.
- Empirical comparison table: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and inference latency (`ms/review`) measured on the exact same unseen test split.

## Slide 5: Visual Evidence & Interpretability
- Confusion Matrices & Score Calibration histograms
- ROC Curves across all 6 models
- Top positive/negative TF-IDF n-grams & live per-token feature attribution
- 2D PCA scatter plot of review semantic space

## Slide 6: Error Analysis & Failure Taxonomy
- Diagnostic audit of False Positives and False Negatives:
  - Concessive contrast (`"good acting, however the plot dragged"`)
  - Long-range negation scope
  - Sarcastic praise & plot-recap dilution
- Concrete cases where DistilBERT's self-attention resolves mistakes made by bag-of-n-grams models.

## Slide 7: Application Security & Live Demo
- Security protocols: CSP (`script-src 'self'`), CSRF tokens, Flask-Limiter rate limiting, file/input validation, OWASP CSV formula injection neutralization, and HSTS/nosniff headers.
- **Live Demo:** Single review analysis (including negation & mixed cases) + Batch CSV upload + Cloud deployment on Render.
