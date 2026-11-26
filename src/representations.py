"""Layer 4: Representation Layer — Bag-of-Words, TF-IDF, and Dense LSA/SVD Embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer


@dataclass
class RepresentationArtifacts:
    """Container for fitted text vectorizers and dense semantic projector."""

    bow_vectorizer: CountVectorizer
    tfidf_vectorizer: TfidfVectorizer
    svd_embedder: TruncatedSVD
    embed_normalizer: Normalizer
    pca_2d: PCA


def build_representations(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
) -> tuple[RepresentationArtifacts, dict[str, object]]:
    """Fit BoW, TF-IDF, and 128-dim SVD dense embeddings on training reviews only (zero leakage)."""
    bow_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        max_features=20000,
    )
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        max_features=25000,
        sublinear_tf=True,
    )

    X_train_bow = bow_vectorizer.fit_transform(train_texts)
    X_val_bow = bow_vectorizer.transform(val_texts)
    X_test_bow = bow_vectorizer.transform(test_texts)

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_val_tfidf = tfidf_vectorizer.transform(val_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)

    # Dense Latent Semantic Embedding representation (128-dim normalized vectors)
    n_components = min(128, X_train_tfidf.shape[1] - 1, X_train_tfidf.shape[0] - 1)
    svd_embedder = TruncatedSVD(n_components=n_components, random_state=42)
    embed_normalizer = Normalizer(copy=False)

    X_train_dense = embed_normalizer.fit_transform(svd_embedder.fit_transform(X_train_tfidf))
    X_val_dense = embed_normalizer.transform(svd_embedder.transform(X_val_tfidf))
    X_test_dense = embed_normalizer.transform(svd_embedder.transform(X_test_tfidf))

    # 2D PCA projection of dense embeddings for interactive dashboard visualization
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d.fit(X_train_dense)
    test_coords_2d = pca_2d.transform(X_test_dense)

    artifacts = RepresentationArtifacts(
        bow_vectorizer=bow_vectorizer,
        tfidf_vectorizer=tfidf_vectorizer,
        svd_embedder=svd_embedder,
        embed_normalizer=embed_normalizer,
        pca_2d=pca_2d,
    )

    matrices = {
        "X_train_bow": X_train_bow,
        "X_val_bow": X_val_bow,
        "X_test_bow": X_test_bow,
        "X_train_tfidf": X_train_tfidf,
        "X_val_tfidf": X_val_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "X_train_dense": X_train_dense,
        "X_val_dense": X_val_dense,
        "X_test_dense": X_test_dense,
        "test_coords_2d": np.asarray(test_coords_2d),
    }

    return artifacts, matrices
