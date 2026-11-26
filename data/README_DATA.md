# Dataset Documentation & Data Dictionary

## 1. Dataset Source
- **Dataset:** Stanford Large Movie Review Dataset (`aclImdb v1.0`)
- **Authors:** Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (ACL 2011)
- **Official URL:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Curated Working Sample:** `data/raw/movie_reviews_imdb.csv` (balanced stratified sample of up to 6,000 real IMDb reviews drawn evenly from official `train` and `test` partitions, deduplicated and verified for non-empty text).
- **Split Files:** `data/processed/train.csv` (70%), `data/processed/val.csv` (15%), `data/processed/test.csv` (15%), stratified by sentiment label with `random_state=42`.

## 2. Data Dictionary

| Column | Type | Description |
| :--- | :--- | :--- |
| `review_id` | string | Unique review identifier (`<partition>_<sentiment>_<id>_<rating>`). |
| `review_text` | string | Raw English IMDb user review text (contains HTML line breaks `<br />` prior to preprocessing). |
| `sentiment` | string | Binary class label (`positive` or `negative`). |
| `label` | integer | Numeric target (`1` = Positive, `0` = Negative). |
| `star_rating_raw` | integer | Original IMDb user rating (`1–4` for negative, `7–10` for positive; neutral 5–6 excluded in Stanford IMDb). |
| `source_partition` | string | Origin folder in Stanford IMDb (`train` or `test`). |
| `dataset_source` | string | Provenance citation string. |
| `char_length` | integer | Character length of the review (computed in `processed` splits). |
| `word_count` | integer | Whitespace token count of the review (computed in `processed` splits). |
