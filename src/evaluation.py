"""Layers 7 & 8: Evaluation & Error Analysis — Comparative metrics, plots, and failure taxonomy."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    ASPECT_KEYWORDS_PATH,
    EMBEDDING_PROJECTION_PATH,
    ERROR_ANALYSIS_PATH,
    EVALUATION_REPORT_PATH,
    PLOTS_DIR,
)
from src.transformer_model import MOVIE_ASPECTS, DistilBertSentimentEngine


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    latency_ms_per_review: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Downsample ROC curve to 25 evenly spaced points for lightweight JSON charting
    indices = np.linspace(0, len(fpr) - 1, min(25, len(fpr)), dtype=int)

    # Probability histogram bins (10 bins from 0.0 to 1.0)
    pos_hist, _ = np.histogram(y_prob[y_true == 1], bins=10, range=(0.0, 1.0))
    neg_hist, _ = np.histogram(y_prob[y_true == 0], bins=10, range=(0.0, 1.0))

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "latency_ms_per_review": round(float(latency_ms_per_review), 3),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[indices]],
            "tpr": [round(float(x), 4) for x in tpr[indices]],
        },
        "score_distribution": {
            "bins": [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
            "true_positive_counts": pos_hist.tolist(),
            "true_negative_counts": neg_hist.tolist(),
        },
    }


def _classify_error_cause(text: str, true_label: int, pred_prob: float) -> tuple[str, str]:
    """Identify probable linguistic or data-related cause for a misclassification."""
    lower = text.lower()
    word_count = len(lower.split())

    has_contrast = bool(re.search(r"\b(but|however|although|though|despite|yet|unfortunately)\b", lower))
    has_negation = bool(re.search(r"\b(not|never|no|hardly|barely|neither|nothing|n't)\b", lower))
    has_sarcasm_cues = bool(re.search(r"(!{2,}|\?\!|\"[^\"]+\"|yeah right|so bad it's good|masterpiece of boredom)", lower))

    if has_sarcasm_cues:
        return (
            "Sarcasm / Irony & Figurative Framing",
            "Contains ironic praise or exaggerated punctuation that reverses literal word polarity.",
        )
    if has_contrast and abs(pred_prob - 0.5) < 0.25:
        return (
            "Mixed / Concessive Review Structure",
            "Combines praise and criticism across contrast markers ('but', 'however', 'despite'), creating balanced polarity cues.",
        )
    if has_negation:
        return (
            "Complex Negation Scope",
            "Negation spans multiple clauses or modifies positive adjectives indirectly, diluting unigram/bigram polarity.",
        )
    if word_count > 320:
        return (
            "Long-Range Plot Summary Dilution",
            "Extended narrative plot recap introduces neutral or dramatic words (e.g., 'murder', 'tragic') unrelated to reviewer verdict.",
        )
    if true_label == 1 and pred_prob < 0.5:
        return (
            "Understated Praise / Genre Vocabulary",
            "Positive review uses dark-genre descriptors (horror/thriller terms) that carry negative prior weights.",
        )
    return (
        "Subtle / Implicit Critique",
        "Negative verdict expressed through neutral facts or polite understatement rather than strong negative adjectives.",
    )


def build_error_analysis(
    test_df: pd.DataFrame,
    logreg_probs: np.ndarray,
    transformer_probs: np.ndarray | None,
) -> dict[str, Any]:
    """Compile real correct, false-positive, false-negative, and model-disagreement examples."""
    records: list[dict[str, Any]] = []
    for idx, row in test_df.reset_index(drop=True).iterrows():
        true_lbl = int(row["label"])
        lr_p = float(logreg_probs[idx])
        lr_pred = 1 if lr_p >= 0.5 else 0
        tr_p = float(transformer_probs[idx]) if transformer_probs is not None else None
        tr_pred = (1 if tr_p >= 0.5 else 0) if tr_p is not None else None
        text = str(row["review_text"])
        cause_tag, explanation = _classify_error_cause(text, true_lbl, lr_p)

        records.append(
            {
                "review_id": str(row["review_id"]),
                "true_sentiment": "positive" if true_lbl == 1 else "negative",
                "true_label": true_lbl,
                "star_rating_raw": int(row["star_rating_raw"]),
                "logreg_prob_pos": round(lr_p, 4),
                "logreg_pred": "positive" if lr_pred == 1 else "negative",
                "logreg_correct": lr_pred == true_lbl,
                "transformer_prob_pos": round(tr_p, 4) if tr_p is not None else None,
                "transformer_pred": (
                    ("positive" if tr_pred == 1 else "negative") if tr_pred is not None else None
                ),
                "transformer_correct": (tr_pred == true_lbl) if tr_pred is not None else None,
                "probable_cause": cause_tag,
                "explanation": explanation,
                "excerpt": re.sub(r"<[^>]+>", " ", text)[:340].strip() + ("..." if len(text) > 340 else ""),
            }
        )

    df_rec = pd.DataFrame(records)
    false_positives = (
        df_rec[(df_rec["true_label"] == 0) & (~df_rec["logreg_correct"])]
        .sort_values("logreg_prob_pos", ascending=False)
        .head(6)
        .to_dict(orient="records")
    )
    false_negatives = (
        df_rec[(df_rec["true_label"] == 1) & (~df_rec["logreg_correct"])]
        .sort_values("logreg_prob_pos", ascending=True)
        .head(6)
        .to_dict(orient="records")
    )
    correct_confident = (
        df_rec[df_rec["logreg_correct"]]
        .assign(conf=lambda d: (d["logreg_prob_pos"] - 0.5).abs())
        .sort_values("conf", ascending=False)
        .drop(columns=["conf"])
        .head(6)
        .to_dict(orient="records")
    )

    transformer_wins: list[dict[str, Any]] = []
    if transformer_probs is not None:
        transformer_wins = (
            df_rec[(~df_rec["logreg_correct"]) & (df_rec["transformer_correct"] == True)]  # noqa: E712
            .head(6)
            .to_dict(orient="records")
        )

    # Error cause frequency across all LogReg mistakes
    mistakes = df_rec[~df_rec["logreg_correct"]]
    cause_breakdown = {
        str(k): int(v) for k, v in mistakes["probable_cause"].value_counts().items()
    }

    payload = {
        "total_test_samples": int(len(df_rec)),
        "logreg_error_count": int(len(mistakes)),
        "transformer_error_count": int((df_rec["transformer_correct"] == False).sum())  # noqa: E712
        if transformer_probs is not None
        else None,
        "cause_breakdown": cause_breakdown,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "correct_examples": correct_confident,
        "transformer_over_classical_wins": transformer_wins,
    }

    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _save_publication_plots(model_results: list[dict[str, Any]]) -> None:
    """Render high-DPI Matplotlib visualizations for report & presentation."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#120D0F",
            "axes.facecolor": "#1A1316",
            "axes.edgecolor": "#46373B",
            "axes.labelcolor": "#F5ECE0",
            "text.color": "#F5ECE0",
            "xtick.color": "#C9B9AA",
            "ytick.color": "#C9B9AA",
        }
    )

    # 1. Model Comparison Bar Chart (Accuracy & F1)
    names = [m["name"] for m in model_results]
    accs = [m["metrics"]["accuracy"] * 100 for m in model_results]
    f1s = [m["metrics"]["f1_score"] * 100 for m in model_results]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    ax.bar(x - width / 2, accs, width, label="Accuracy (%)", color="#E5B25D")
    ax.bar(x + width / 2, f1s, width, label="F1-Score (%)", color="#52C79A")
    ax.set_ylabel("Score (%)")
    ax.set_title("Classical Baselines vs. DistilBERT Transformer — Test Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=16, ha="right")
    ax.set_ylim(60, 100)
    ax.grid(axis="y", alpha=0.18, color="#E5B25D")
    ax.legend(frameon=False)
    for idx, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(idx - width / 2, a + 0.6, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(idx + width / 2, f + 0.6, f"{f:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png")
    plt.close(fig)

    # 2. ROC Curves
    palette = ["#D9534F", "#E07A5F", "#E5B25D", "#F4CE85", "#9B72AA", "#52C79A"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    for idx, m in enumerate(model_results):
        roc = m["metrics"]["roc_curve"]
        ax.plot(
            roc["fpr"],
            roc["tpr"],
            linewidth=2.1,
            color=palette[idx % len(palette)],
            label=f"{m['name']} (AUC={m['metrics']['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6D5C61", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Across Text Representations & Models")
    ax.grid(alpha=0.15, color="#E5B25D")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png")
    plt.close(fig)

    # 3. Confusion Matrix Comparison (Best Classical vs Transformer)
    selected = [m for m in model_results if m["key"] in {"tfidf_logreg", "distilbert_onnx"}]
    if selected:
        fig, axes = plt.subplots(1, len(selected), figsize=(5.2 * len(selected), 4.4), dpi=180)
        if len(selected) == 1:
            axes = [axes]
        for ax, m in zip(axes, selected):
            cm = np.array(m["metrics"]["confusion_matrix"])
            ax.imshow(cm, cmap="YlOrBr")
            ax.set_title(m["name"], fontsize=10.5)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Neg", "Pred Pos"])
            ax.set_yticklabels(["True Neg", "True Pos"])
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    color = "#120D0F" if val > cm.max() * 0.55 else "#F5ECE0"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "confusion_matrices.png")
        plt.close(fig)


def evaluate_all_models(
    test_df: pd.DataFrame,
    matrices: dict[str, Any],
    trained_classical: dict[str, dict[str, Any]],
    top_ngrams: dict[str, Any],
    transformer_engine: DistilBertSentimentEngine,
) -> dict[str, Any]:
    """Run unified test-set benchmarking across classical and transformer models."""
    y_test = test_df["label"].to_numpy(dtype=int)
    model_results: list[dict[str, Any]] = []

    feature_lookup = {
        "bow": matrices["X_test_bow"],
        "tfidf": matrices["X_test_tfidf"],
        "dense": matrices["X_test_dense"],
    }

    logreg_probs = None
    for key, info in trained_classical.items():
        X_test = feature_lookup[info["feature_space"]]
        est = info["estimator"]
        t0 = time.perf_counter()
        probs = est.predict_proba(X_test)[:, 1]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ms_per_review = elapsed_ms / max(len(y_test), 1)

        if key == "tfidf_logreg":
            logreg_probs = probs

        model_results.append(
            {
                "key": key,
                "name": info["name"],
                "family": "Classical ML",
                "representation": info["representation"],
                "train_time_ms": info["train_time_ms"],
                "val_accuracy": info["val_accuracy"],
                "metrics": _compute_metrics(y_test, probs, ms_per_review),
            }
        )

    # Evaluate DistilBERT ONNX Transformer on exact same test split
    transformer_probs: np.ndarray | None = None
    if transformer_engine.load():
        raw_test_texts = [re.sub(r"<[^>]+>", " ", t) for t in test_df["review_text"].astype(str).tolist()]
        transformer_probs, total_ms = transformer_engine.predict_batch(raw_test_texts, batch_size=32)
        ms_per_review = total_ms / max(len(y_test), 1)
        model_results.append(
            {
                "key": "distilbert_onnx",
                "name": "DistilBERT (SST-2 ONNX)",
                "family": "Pretrained Transformer",
                "representation": "Contextual Subword Embeddings (768-dim)",
                "train_time_ms": None,
                "val_accuracy": None,
                "metrics": _compute_metrics(y_test, transformer_probs, ms_per_review),
            }
        )

    # Save 2D Embedding Projection (sample up to 450 test points for snappy scatter plot)
    coords = matrices["test_coords_2d"]
    sample_n = min(450, len(test_df))
    proj_points = []
    for i in range(sample_n):
        row = test_df.iloc[i]
        proj_points.append(
            {
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "sentiment": str(row["sentiment"]),
                "rating": int(row["star_rating_raw"]),
                "excerpt": re.sub(r"<[^>]+>", " ", str(row["review_text"]))[:110].strip() + "...",
            }
        )
    with open(EMBEDDING_PROJECTION_PATH, "w", encoding="utf-8") as f:
        json.dump({"points": proj_points}, f, indent=2)

    # Corpus-level Movie Aspect Distribution on Test Split
    aspect_stats = []
    test_lower = test_df["review_text"].astype(str).str.lower()
    for aspect_name, kw_set in MOVIE_ASPECTS.items():
        pattern = r"\b(" + "|".join(sorted(kw_set)) + r")\b"
        mask = test_lower.str.contains(pattern, regex=True)
        subset = test_df[mask]
        total_mentions = int(len(subset))
        pos_rate = round(float(subset["label"].mean()), 4) if total_mentions > 0 else 0.5
        aspect_stats.append(
            {
                "aspect": aspect_name,
                "review_mentions": total_mentions,
                "share_pct": round(100.0 * total_mentions / max(len(test_df), 1), 1),
                "positive_rate": pos_rate,
            }
        )
    with open(ASPECT_KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"aspects": aspect_stats}, f, indent=2)

    best_model = max(model_results, key=lambda m: m["metrics"]["f1_score"])
    report = {
        "test_set_size": int(len(y_test)),
        "best_model_key": best_model["key"],
        "best_model_name": best_model["name"],
        "models": model_results,
        "top_ngrams": top_ngrams,
    }
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if logreg_probs is not None:
        build_error_analysis(test_df, logreg_probs, transformer_probs)

    _save_publication_plots(model_results)
    return report
