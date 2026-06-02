"""
Compare complexity_v1 (standardized mean) vs complexity_v2 (PCA axis)
for the Cognitive Complexity Gradient Mapper.

Reads:
  data/processed/para_features.csv
  data/processed/article_features.csv

Outputs:
  outputs/complexity_comparison/01_v1_v2_scatter.png
  outputs/complexity_comparison/02_length_proxy_check.png
  outputs/complexity_comparison/03_top_bottom_paragraphs.txt
  outputs/complexity_comparison/04_cv_comparison.csv

Run: python scripts/compare_complexity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from article_features import add_complexity_scores

PARA_PATH    = Path("data/processed/para_features.csv")
ARTICLE_PATH = Path("data/processed/article_features.csv")
OUT_DIR      = Path("outputs/complexity_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_paragraphs() -> pd.DataFrame:
    para = (
        pd.read_csv(PARA_PATH)
        .sort_values(["post_url", "paragraph_index"])
        .reset_index(drop=True)
    )
    print(f"Loaded {len(para)} paragraphs across {para['post_url'].nunique()} articles")
    return add_complexity_scores(para)


# ── 1. Scatter: v1 vs v2 ──────────────────────────────────────────────────────

def plot_scatter(para: pd.DataFrame) -> None:
    v1 = para["complexity_v1"].values
    v2 = para["complexity_v2"].values
    pearson_r, _ = pearsonr(v1, v2)
    spearman_r, _ = spearmanr(v1, v2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(v1, v2, alpha=0.15, s=8, rasterized=True)
    ax.set_xlabel("complexity_v1  (standardized mean)")
    ax.set_ylabel("complexity_v2  (PCA first component)")
    ax.set_title(f"v1 vs v2  —  Pearson r={pearson_r:.3f}  Spearman ρ={spearman_r:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_v1_v2_scatter.png", dpi=150)
    plt.close(fig)

    print(f"\n[1] v1 vs v2  Pearson r={pearson_r:.3f}  Spearman ρ={spearman_r:.3f}")


# ── 2. Length-proxy check ─────────────────────────────────────────────────────

def plot_length_proxy(para: pd.DataFrame) -> None:
    proxy_cols = ["word_count", "fk_grade", "avg_sentence_length", "ner_count"]
    available  = [c for c in proxy_cols if c in para.columns]

    rows = []
    for col in available:
        subset = para[["complexity_v1", "complexity_v2", col]].dropna()
        for score in ("complexity_v1", "complexity_v2"):
            r, _ = spearmanr(subset[score], subset[col])
            rows.append({"feature": col, "score": score, "spearman_rho": round(r, 3)})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="feature", columns="score", values="spearman_rho")
    print("\n[2] Spearman ρ with proxy features:")
    print(pivot.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax, width=0.6)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Correlation with Proxy Features  (length-proxy check)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_length_proxy_check.png", dpi=150)
    plt.close(fig)


# ── 3. Top / bottom paragraph examples ───────────────────────────────────────

def print_examples(para: pd.DataFrame, n: int = 5) -> None:
    lines = []
    for score in ("complexity_v1", "complexity_v2"):
        ranked = para.dropna(subset=[score, "paragraph_text"]).sort_values(score)

        lines.append(f"\n{'='*60}")
        lines.append(f"  {score}  —  top {n} most complex")
        lines.append("=" * 60)
        for _, row in ranked.tail(n).iloc[::-1].iterrows():
            snippet = str(row["paragraph_text"])[:120].replace("\n", " ")
            lines.append(f"  [{score}={row[score]:+.3f}]  {snippet}...")

        lines.append(f"\n  {score}  —  bottom {n} least complex")
        lines.append("-" * 60)
        for _, row in ranked.head(n).iterrows():
            snippet = str(row["paragraph_text"])[:120].replace("\n", " ")
            lines.append(f"  [{score}={row[score]:+.3f}]  {snippet}...")

    output = "\n".join(lines)
    print(output)
    (OUT_DIR / "03_top_bottom_paragraphs.txt").write_text(output, encoding="utf-8")


# ── 4. Engagement CV comparison ───────────────────────────────────────────────

def cv_compare_engagement(para: pd.DataFrame) -> None:
    articles = pd.read_csv(ARTICLE_PATH)

    means = (
        para.groupby("post_url")[["complexity_v1", "complexity_v2"]]
        .mean()
        .rename(columns={"complexity_v1": "mean_v1", "complexity_v2": "mean_v2"})
        .reset_index()
    )
    df = articles.merge(means, on="post_url", how="inner")
    df = df.dropna(
        subset=["engagement_z", "above_median_engagement_by_publication", "source_key", "mean_v1", "mean_v2"]
    )

    if len(df) < 10:
        print("\n[4] Too few articles with engagement data for CV — skipping")
        return

    n_sources = df["source_key"].nunique()
    print(f"\n[4] CV comparison on {len(df)} articles across {n_sources} publications")

    groups = df["source_key"].values
    y_reg  = df["engagement_z"].values.astype(float)
    y_clf  = df["above_median_engagement_by_publication"].values.astype(int)
    gkf    = GroupKFold(n_splits=min(5, n_sources))

    rows = []
    for label, feat_col in [("mean_v1", "mean_v1"), ("mean_v2", "mean_v2")]:
        X = df[[feat_col]].values

        reg_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        clf_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))])

        reg_preds, reg_trues       = [], []
        clf_preds, clf_probs, clf_trues = [], [], []

        for train_idx, test_idx in gkf.split(X, y_reg, groups):
            reg_pipe.fit(X[train_idx], y_reg[train_idx])
            reg_preds.extend(reg_pipe.predict(X[test_idx]))
            reg_trues.extend(y_reg[test_idx])

            if len(np.unique(y_clf[train_idx])) < 2:
                continue
            clf_pipe.fit(X[train_idx], y_clf[train_idx])
            clf_preds.extend(clf_pipe.predict(X[test_idx]))
            clf_probs.extend(clf_pipe.predict_proba(X[test_idx])[:, 1])
            clf_trues.extend(y_clf[test_idx])

        row: dict = {
            "score":    label,
            "MAE":      round(mean_absolute_error(reg_trues, reg_preds), 4),
            "RMSE":     round(root_mean_squared_error(reg_trues, reg_preds), 4),
            "Spearman": round(float(spearmanr(reg_trues, reg_preds).statistic), 4),
        }
        if clf_trues:
            row["F1"]      = round(f1_score(clf_trues, clf_preds, zero_division=0), 4)
            row["ROC-AUC"] = round(roc_auc_score(clf_trues, clf_probs), 4)
        rows.append(row)

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    result.to_csv(OUT_DIR / "04_cv_comparison.csv", index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    para = load_paragraphs()
    plot_scatter(para)
    plot_length_proxy(para)
    print_examples(para)
    cv_compare_engagement(para)
    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
