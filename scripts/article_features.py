"""
Article-level gradient feature extraction for the Cognitive Complexity Gradient Mapper.
Input:  data/processed/para_features.csv
        data/processed/posts_scraped.csv
Output: data/processed/article_features.csv
Run:    python article_features.py  (requires para_features.py to have run first)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PARA_PATH  = Path("data/processed/para_features.csv")
POSTS_PATH = Path("data/processed/posts_scraped.csv")
OUT_PATH   = Path("data/processed/article_features.csv")

# Features pooled across all paragraphs to build the composite complexity score.
# Higher values of each indicate greater cognitive load.
COMPLEXITY_FEATURES = [
    "fk_grade",           # syntactic difficulty
    "avg_sentence_length", # sentence complexity
    "complex_word_rate",  # polysyllabic density
    "type_token_ratio",   # lexical variety
    "ner_count",          # concept density
    "new_entity_count",   # concept introduction rate
    "embed_dist_prev",    # inferential gap from previous paragraph
]


# ── Complexity scores ─────────────────────────────────────────────────────────

def add_complexity_scores(para: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two complexity columns to the paragraph dataframe:
      complexity_v1 — standardized mean of core features (heuristic baseline)
      complexity_v2 — first PCA component (data-driven axis)
    Standardization is across all paragraphs, not per article.
    """
    available = [f for f in COMPLEXITY_FEATURES if f in para.columns]
    X = para[available].fillna(0)

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available, index=para.index)

    para = para.copy()
    para["complexity_v1"] = X_scaled.mean(axis=1)

    pca = PCA(n_components=1)
    para["complexity_v2"] = pca.fit_transform(X_scaled).flatten()

    loadings = pd.Series(pca.components_[0], index=available).sort_values(ascending=False)
    print(f"complexity_v2 PCA — variance explained: {pca.explained_variance_ratio_[0]:.1%}")
    print(loadings.round(3).to_string(), "\n")
    return para


# ── Gradient features ─────────────────────────────────────────────────────────

def _gradient_features(scores: np.ndarray) -> dict:
    """Derive shape and trajectory features from a paragraph complexity sequence."""
    n  = len(scores)
    t1 = max(1, round(n * 0.2))   # boundary of opening section (~20%)
    t2 = min(n - 1, round(n * 0.8))  # boundary of ending section (~20%)

    opening = float(scores[:t1].mean())
    middle  = float(scores[t1:t2].mean()) if t1 < t2 else opening
    ending  = float(scores[t2:].mean())   if t2 < n  else float(scores[-1])

    diffs     = np.diff(scores) if n > 1 else np.array([0.0])
    slope     = float(np.polyfit(np.arange(n, dtype=float), scores, 1)[0]) if n > 1 else 0.0
    mean_s    = float(scores.mean())
    std_s     = float(scores.std())
    max_s     = float(scores.max())
    min_s     = float(scores.min())
    peak_idx  = int(scores.argmax())

    return {
        "mean_complexity":     mean_s,
        "max_complexity":      max_s,
        "complexity_variance": float(np.var(scores)),
        "complexity_range":    max_s - min_s,
        "complexity_slope":    slope,
        "opening_complexity":  opening,
        "middle_complexity":   middle,
        "ending_complexity":   ending,
        "peak_position":       peak_idx / max(n - 1, 1),
        "resolution_index":    max_s - ending,
        "jumpiness":           float(np.var(diffs)),
        "derivative_std":      float(np.std(diffs)),
        "n_spikes":            int((scores > mean_s + std_s).sum()),
        "above_mean_rate":     float((scores > mean_s).mean()),
        "early_spike_score":   float(scores[:t1].max()),
    }


def _embed_gradient(group: pd.DataFrame) -> dict:
    """Gradient features derived from paragraph-to-paragraph embedding distances."""
    if "embed_dist_prev" not in group.columns:
        return {}
    dists = group["embed_dist_prev"].values
    n     = len(dists)
    return {
        "mean_embed_dist":  float(dists.mean()),
        "max_embed_dist":   float(dists.max()),
        "embed_dist_slope": float(np.polyfit(np.arange(n), dists, 1)[0]) if n > 1 else 0.0,
    }


def _sentiment_gradient(group: pd.DataFrame) -> dict:
    """
    Article-level sentiment features derived from paragraph VADER scores.
    Grounded in two cognitive load mechanisms:
      - Polarity load: negative sentiment increases attentional demand.
      - Coherence load: abrupt valence shifts increase extraneous cognitive load.
    Returns an empty dict if VADER columns are absent (backwards-compatible).
    """
    if "vader_compound" not in group.columns or "vader_neg" not in group.columns:
        return {}

    compound = group["vader_compound"].values.astype(float)
    neg      = group["vader_neg"].values.astype(float)
    n        = len(compound)

    # Polarity sign per VADER standard thresholds: +1 positive, -1 negative, 0 neutral
    signs = np.where(compound > 0.05, 1, np.where(compound < -0.05, -1, 0))

    if n > 1:
        flips          = int(np.sum((signs[:-1] != signs[1:]) & (signs[:-1] != 0) & (signs[1:] != 0)))
        shift_rate     = flips / (n - 1)
        mean_abs_shift = float(np.abs(np.diff(compound)).mean())
        slope          = float(np.polyfit(np.arange(n, dtype=float), compound, 1)[0])
    else:
        shift_rate     = 0.0
        mean_abs_shift = 0.0
        slope          = 0.0

    return {
        "mean_sentiment_compound":  float(compound.mean()),
        "mean_sentiment_neg":       float(neg.mean()),
        "negative_paragraph_rate":  float((compound < -0.05).mean()),
        "sentiment_variance":       float(compound.var()),
        "mean_abs_sentiment_shift": mean_abs_shift,
        "sentiment_shift_rate":     shift_rate,
        "sentiment_slope":          slope,
    }


# ── Gradient shape label ──────────────────────────────────────────────────────

def _label_shape(
    row: pd.Series,
    jump_thresh: float,
    var_thresh: float,
    slope_thresh: float,
) -> str:
    """
    Rule-based gradient shape label. Priority order:
      rollercoaster → plateau → ramp → cliff → resolution
    Thresholds are calibrated from the dataset (passed in from main).
    """
    if row["jumpiness"]           >= jump_thresh:  return "rollercoaster"
    if row["complexity_variance"] <= var_thresh:   return "plateau"
    if row["complexity_slope"]    >= slope_thresh: return "ramp"
    if row["peak_position"]       <= 0.3:          return "cliff"
    return "resolution"


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    para  = pd.read_csv(PARA_PATH).sort_values(["post_url", "paragraph_index"]).reset_index(drop=True)
    posts = pd.read_csv(POSTS_PATH)

    print(f"Input: {len(para)} paragraph rows across {para['post_url'].nunique()} articles\n")
    para = add_complexity_scores(para)

    rows = []
    for post_url, group in para.groupby("post_url"):
        scores = group["complexity_v1"].values
        feat   = {"post_url": post_url, "n_paragraphs": len(group)}
        feat.update(_gradient_features(scores))
        feat.update(_embed_gradient(group))
        feat.update(_sentiment_gradient(group))
        rows.append(feat)

    feats = pd.DataFrame(rows)

    # Calibrate shape thresholds from the full dataset distribution
    jump_thresh  = float(feats["jumpiness"].quantile(0.75))
    var_thresh   = float(feats["complexity_variance"].quantile(0.25))
    slope_thresh = float(feats["complexity_slope"].std())

    feats["gradient_shape"] = feats.apply(
        _label_shape, axis=1,
        jump_thresh=jump_thresh, var_thresh=var_thresh, slope_thresh=slope_thresh,
    )
    print("Gradient shape distribution:")
    print(feats["gradient_shape"].value_counts().to_string(), "\n")

    meta_cols = [
        "post_url", "publication_url", "publication_name", "source_key", "publish_date",
        "title", "word_count", "paragraph_count",
        "like_count", "comment_count", "engagement_available",
    ]
    meta = posts[[c for c in meta_cols if c in posts.columns]]
    out  = meta.merge(feats, on="post_url", how="right")

    out["engagement_raw"] = np.log1p(
        out["like_count"].fillna(0) + 2 * out["comment_count"].fillna(0)
    )
    grp = out.groupby("source_key")["engagement_raw"]
    out["engagement_z"] = (out["engagement_raw"] - grp.transform("mean")) / grp.transform("std")
    out["above_median_engagement_by_publication"] = (
        out["engagement_raw"] >= grp.transform("median")
    ).astype(int)

    out.to_csv(OUT_PATH, index=False)
    print(f"Output: {len(out)} rows × {out.shape[1]} columns → {OUT_PATH}")


if __name__ == "__main__":
    main()
