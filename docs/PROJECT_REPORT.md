# ETMMTI303 Capstone Project Report
## Option 12: CinePulse NLP — Movie Review Sentiment Dashboard

**Course:** ETMMTI303 – Language Technologies and Text Intelligence (MCA AI & ML, Semester III)  
**Institution:** K.R. Mangalam University, School of Engineering and Technology  

---

## 1. Problem Definition
Streaming platforms, film distributors, and audience analytics teams receive thousands of unstructured movie reviews daily. Manual moderation cannot scale to track audience reception, polarity shifts, or aspect-level feedback (acting, plot, direction, visuals, soundtrack).

- **Target Users:** Film analysts, content curators, cinema researchers, and NLP evaluators.
- **Expected Input:** Unseen single movie reviews (up to 10,000 characters) or batch `.csv`/`.txt` review collections (up to 500 reviews per upload).
- **Expected Output:** Binary sentiment polarity (`Positive` / `Negative`), calibrated confidence score, estimated IMDb-style 1–10 rating, token-level attribution, movie aspect sentiment breakdown, preprocessing audit trace, and comparative model metrics.

---

## 2. Dataset Description & Splitting Strategy
- **Source:** Stanford Large Movie Review Dataset (`aclImdb v1.0`, Maas et al., ACL 2011 — `https://ai.stanford.edu/~amaas/data/sentiment/`).
- **Corpus Size:** `6,000` raw balanced reviews sampled evenly from official train/test folders; `12` exact/normalized duplicates removed, yielding **`5,988` clean reviews** (`2,993` positive, `2,995` negative).
- **Length Statistics:** Mean `232.6` words, median `173.0` words (range `8` to `2,470` words).
- **Leakage-Free Stratified Splits (`random_state=42`):**
  - **Train:** `4,191` reviews (70%)
  - **Validation:** `898` reviews (15%)
  - **Test (Held-Out Unseen):** `899` reviews (15%)

---

## 3. Preprocessing Pipeline (`src/preprocessing.py`)
1. **HTML & Noise Cleanup:** Unescape HTML entities and remove `<br />` tags, residual HTML, and URLs.
2. **Contraction Expansion:** Expand negations (`don't` → `do not`, `wasn't` → `was not`, `can't` → `cannot`) prior to punctuation stripping.
3. **Negation-Aware Stop-Word Policy:** Standard English stop words are removed, **excluding** negation and contrast markers (`no`, `nor`, `not`, `never`, `neither`, `hardly`, `barely`, `but`, `however`, `although`, `though`, `yet`, `very`, `too`), which are crucial for sentiment inversion.
4. **WordNet Lemmatization:** Tokens are normalized to canonical dictionary lemmas via NLTK `WordNetLemmatizer`.

---

## 4. Text Representations (`src/representations.py`)
All vectorizers are fitted strictly on the training split:
1. **Bag-of-Words (`CountVectorizer`):** Unigrams and bigrams (`ngram_range=(1,2)`, `min_df=3`, capped at 20,000 features).
2. **Sublinear TF-IDF (`TfidfVectorizer`):** Sublinear term-frequency scaling (`1 + log(tf)`), unigrams & bigrams (`ngram_range=(1,2)`, `min_df=3`, capped at 25,000 features).
3. **Dense Latent Semantic Embeddings (`TruncatedSVD` + L2 Normalizer):** 128-dimensional dense semantic vectors derived from TF-IDF, projected to 2D via PCA for cluster visualization.
4. **Contextual Transformer Subword Embeddings:** 768-dimensional DistilBERT hidden representations via ONNX Runtime.

---

## 5. Modeling Architecture (Classical + Transformer)
- **Classical Models (`src/classical_models.py`):**
  - BoW + Multinomial Naive Bayes (`alpha=0.5`)
  - TF-IDF + Multinomial Naive Bayes (`alpha=0.4`)
  - TF-IDF + Logistic Regression (`C=2.0`)
  - TF-IDF + Calibrated Linear SVM (`LinearSVC(C=0.8)` + 3-fold sigmoid calibration)
  - Dense 128-dim SVD Embeddings + Logistic Regression (`C=1.5`)
- **Transformer Component (`src/transformer_model.py`):**
  - Quantized **DistilBERT SST-2 (`Xenova/distilbert-base-uncased-finetuned-sst-2-english`)** executed via `onnxruntime` + Rust `tokenizers` for CPU-efficient cloud deployment (head+tail chunking for long reviews during interactive inference).

---

## 6. Evaluation & Comparative Benchmark (`n = 899` Unseen Test Reviews)

| Model | Representation | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Latency (ms/rev) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoW + Multinomial NB** | Bag-of-Words (1–2g) | 87.65% | 89.67% | 85.08% | 0.8731 | 0.9387 | 0.002 |
| **TF-IDF + Multinomial NB** | Sublinear TF-IDF (1–2g) | 88.88% | 91.06% | 86.19% | 0.8856 | 0.9527 | 0.001 |
| **TF-IDF + Logistic Regression** | Sublinear TF-IDF (1–2g) | 89.43% | 90.23% | 88.42% | 0.8931 | 0.9592 | 0.001 |
| **TF-IDF + Linear SVM** | Sublinear TF-IDF (1–2g) | **89.99%** | **91.84%** | 87.75% | **0.8975** | **0.9623** | 0.004 |
| **Dense SVD + LogReg** | LSA Dense (128-dim) | 87.88% | 91.26% | 83.74% | 0.8734 | 0.9477 | <0.001 |
| **DistilBERT (SST-2 ONNX)** | Contextual Subword (768d) | 87.76% | 89.33% | 85.75% | 0.8750 | 0.9472 | 157.31 |

### Key Empirical Finding (Supporting Claims with Data)
On full-length IMDb reviews (mean 232.6 words), **in-domain `TF-IDF + Linear SVM` (89.99% accuracy, 0.9623 ROC-AUC)** outperformed zero-shot `DistilBERT SST-2` (87.76% accuracy, 0.9472 ROC-AUC) while executing ~39,000x faster on CPU. Two factors explain this result:
1. **Domain & Length Shift:** SST-2 DistilBERT is fine-tuned on short single-sentence Rotten Tomatoes snippets, whereas IMDb reviews frequently open with neutral plot recaps before stating the verdict in closing paragraphs (truncated at 256 subword tokens in batch mode).
2. **In-Domain Lexical Discrimination:** Bigram TF-IDF directly captures IMDb-specific idioms (`"one best"` +2.60, `"worst"` -4.56, `"waste"` -3.37).
However, qualitative error inspection shows DistilBERT resolves several subtle concessive and negation-shift errors where linear n-grams fail, motivating our weighted **CinePulse Ensemble** (`0.60 * DistilBERT + 0.25 * LogReg + 0.15 * SVM`) with head+tail chunking for interactive single-review analysis.

Visual evidence is exported to `artifacts/plots/model_comparison.png`, `artifacts/plots/roc_curves.png`, and `artifacts/plots/confusion_matrices.png`.

---

## 7. Error Analysis (`artifacts/error_analysis.json`)
Across 899 test reviews, `TF-IDF + Logistic Regression` produced `95` misclassifications (`43` False Positives, `52` False Negatives), while `DistilBERT SST-2` produced `110` errors (`46` FP, `64` FN). Diagnostic breakdown of LogReg errors:
- **Sarcasm / Irony & Figurative Framing:** 44 cases (e.g., exaggerated praise used ironically in 1-star reviews).
- **Mixed / Concessive Review Structure:** 36 cases (reviews praising acting/visuals while condemning plot via `"but"` / `"however"`).
- **Complex Negation Scope:** 12 cases (multi-clause negation diluting positive adjectives).
- **Subtle / Implicit Critique & Genre Vocabulary:** 3 cases (e.g., polite factual disappointment like *"this picture was far from great"* — where DistilBERT correctly predicted Negative at 33.0% positivity while LogReg predicted 92.8% Positive).

---

## 8. Security Architecture (`src/security.py`)
- Strict `Content-Security-Policy` (`script-src 'self'` with locally vendored Chart.js), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `Permissions-Policy`, and `HSTS` in production.
- Per-session cryptographic CSRF tokens verified via constant-time `hmac.compare_digest` plus `Origin` validation on all POST endpoints.
- Rate limiting (`Flask-Limiter`), 10,000-char text cap, 2 MB / 500-row upload cap, null-byte rejection, DOM `textContent` XSS defense, and OWASP CSV formula-injection escaping (`=`, `+`, `-`, `@`).

---

## 9. Limitations & Future Work
1. **Fine-Tuning DistilBERT on IMDb:** Fine-tuning DistilBERT directly on the IMDb training split (with 512-token longformer/head-tail pooling) would close the domain-shift gap.
2. **Neutral / 5–6 Star Handling:** Stanford IMDb excludes 5–6 star reviews; adding a calibrated 3-class (Positive / Mixed / Negative) corpus such as Rotten Tomatoes multi-class reviews would refine borderline predictions.
3. **Multilingual Support:** Extending preprocessing and mBERT/XLM-R embeddings for non-English cinema reviews.

---

## 10. References
1. Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL 2011.
2. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS EMC2 Workshop.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12.
4. OWASP Cheat Sheet Series: *Cross-Site Request Forgery Prevention*, *XSS Prevention*, and *CSV Injection*.
