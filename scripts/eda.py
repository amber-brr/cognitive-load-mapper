"""
EDA for Cognitive Complexity Gradient Mapper.
Reads data/processed/ CSVs; saves figures to outputs/eda/.
Run: python eda.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", message="An input array is constant")

DATA_DIR = Path("data/processed")
OUT_DIR = Path("outputs/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Load ──────────────────────────────────────────────────────────────────────

pubs    = pd.read_csv(DATA_DIR / "publications_verified.csv")
posts   = pd.read_csv(DATA_DIR / "posts_scraped.csv", parse_dates=["publish_date"])
paras   = pd.read_csv(DATA_DIR / "paragraphs.csv")
summary = pd.read_csv("outputs/scrape_summary.csv").iloc[0]

success = posts[posts["scrape_status"] == "success"].copy()
success["pub_label"] = success["source_key"].str.replace("_", " ").str.title()


# ── Console summary ───────────────────────────────────────────────────────────

n_found   = int(summary["posts_found"])
n_ok      = int(summary["posts_scraped_success"])
n_failed  = int(summary["posts_failed"])
n_eng     = int(summary["posts_with_engagement"])
n_paras   = int(summary["paragraphs_total"])

print("=== Pipeline Summary ===")
print(f"  Posts found (indexed):   {n_found:>5}")
print(f"  Scraped successfully:    {n_ok:>5}  ({n_ok / n_found * 100:.1f}% of found)")
print(f"  Scrape failures:         {n_failed:>5}  ({n_failed / (n_ok + n_failed) * 100:.1f}% failure rate)")
print(f"  With engagement metrics: {n_eng:>5}  ({n_eng / n_ok * 100:.1f}% of scraped OK)")
print(f"  Total paragraphs:        {n_paras:>5}  (avg {n_paras / n_ok:.1f} per article)")

print("\nScrape failure breakdown:")
print(posts["scrape_status"].value_counts().to_string())

print("\nEngagement source breakdown (successful posts):")
print(success["engagement_source"].value_counts().to_string())


# ── 1. Scrape pipeline funnel ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 3))
stages = ["Posts found", "Scraped OK", "With engagement"]
counts = [n_found, n_ok, n_eng]
colors = sns.color_palette("muted", 3)
bars = ax.barh(stages, counts, color=colors)
for bar, n in zip(bars, counts):
    ax.text(bar.get_width() + n_found * 0.01, bar.get_y() + bar.get_height() / 2,
            str(n), va="center", fontsize=11)
ax.set_xlabel("Count")
ax.set_title("Scrape Pipeline Funnel")
ax.set_xlim(0, n_found * 1.15)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_pipeline_funnel.png", dpi=150)
plt.close(fig)


# ── 2. Per-publication breakdown ──────────────────────────────────────────────

pub_stats = (
    success.groupby("pub_label")
    .agg(
        posts=("post_url", "count"),
        with_engagement=("engagement_available", "sum"),
        median_words=("word_count", "median"),
        median_paragraphs=("paragraph_count", "median"),
    )
    .sort_values("posts", ascending=False)
)
print("\n=== Per-Publication Summary ===")
print(pub_stats.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
pub_stats[["posts", "with_engagement"]].plot(kind="bar", ax=ax, width=0.65)
ax.set_ylabel("Post count")
ax.set_title("Posts per Publication")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(["Total scraped", "With engagement"])
fig.tight_layout()
fig.savefig(OUT_DIR / "02_posts_per_publication.png", dpi=150)
plt.close(fig)


# ── 3. Article length distributions ──────────────────────────────────────────

wc = success["word_count"].dropna()
pc = success["paragraph_count"].dropna()

print(f"\n=== Article Length ===")
print(f"  Word count:      min={wc.min():.0f}  median={wc.median():.0f}  p95={wc.quantile(0.95):.0f}  max={wc.max():.0f}")
print(f"  Paragraph count: min={pc.min():.0f}  median={pc.median():.0f}  p95={pc.quantile(0.95):.0f}  max={pc.max():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pal = sns.color_palette("muted")

axes[0].hist(wc, bins=40, edgecolor="white", color=pal[0])
axes[0].axvline(wc.median(), color="red", linestyle="--", label=f"Median {wc.median():.0f}")
axes[0].set_xlabel("Word count")
axes[0].set_ylabel("Articles")
axes[0].set_title("Article Word Count")
axes[0].legend()

axes[1].hist(pc, bins=30, edgecolor="white", color=pal[1])
axes[1].axvline(pc.median(), color="red", linestyle="--", label=f"Median {pc.median():.0f}")
axes[1].set_xlabel("Paragraph count")
axes[1].set_ylabel("Articles")
axes[1].set_title("Article Paragraph Count")
axes[1].legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "03_article_length_distributions.png", dpi=150)
plt.close(fig)


# ── 4. Engagement distributions ──────────────────────────────────────────────

eng = success[success["engagement_available"]].copy()
eng["log_likes"]    = np.log1p(eng["like_count"].fillna(0))
eng["log_comments"] = np.log1p(eng["comment_count"].fillna(0))
eng["engagement_raw"] = np.log1p(eng["like_count"].fillna(0) + 2 * eng["comment_count"].fillna(0))

like_nan_pct    = eng["like_count"].isna().mean() * 100
comment_nan_pct = eng["comment_count"].isna().mean() * 100
eng_both = eng.dropna(subset=["like_count", "comment_count"])
spearman_r = eng_both["like_count"].corr(eng_both["comment_count"], method="spearman") if len(eng_both) > 1 else float("nan")

print(f"\n=== Engagement ===")
print(f"  like_count missing:    {like_nan_pct:.1f}%")
print(f"  comment_count missing: {comment_nan_pct:.1f}%")
print(f"  Spearman(likes, comments) = {spearman_r:.3f}")
print(f"  engagement_raw: median={eng['engagement_raw'].median():.2f}  std={eng['engagement_raw'].std():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(eng["log_likes"], bins=30, edgecolor="white", color=pal[2])
axes[0].set_xlabel("log(1 + likes)")
axes[0].set_title(f"Likes  (NaN={like_nan_pct:.0f}%)")

axes[1].hist(eng["log_comments"], bins=30, edgecolor="white", color=pal[3])
axes[1].set_xlabel("log(1 + comments)")
axes[1].set_title(f"Comments  (NaN={comment_nan_pct:.0f}%)")

axes[2].scatter(eng["log_likes"], eng["log_comments"], alpha=0.3, s=20, color=pal[4])
axes[2].set_xlabel("log(1 + likes)")
axes[2].set_ylabel("log(1 + comments)")
axes[2].set_title(f"Likes vs Comments  (ρ={spearman_r:.2f})")

fig.tight_layout()
fig.savefig(OUT_DIR / "04_engagement_distributions.png", dpi=150)
plt.close(fig)


# ── 5. Engagement by publication ──────────────────────────────────────────────

eng_pub = (
    eng.groupby("pub_label")["engagement_raw"]
    .agg(["median", "std"])
    .sort_values("median", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 4))
eng_pub["median"].plot(kind="bar", yerr=eng_pub["std"], ax=ax,
                       capsize=4, width=0.6, color=pal[0], error_kw={"elinewidth": 1.2})
ax.set_ylabel("log(1 + likes + 2·comments)")
ax.set_title("Median Engagement Score by Publication  (±1 std)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(OUT_DIR / "05_engagement_by_publication.png", dpi=150)
plt.close(fig)


# ── 6. Paragraph-level distributions ──────────────────────────────────────────

paras_per_article = paras.groupby("post_url").size()

print(f"\n=== Paragraphs ===")
print(f"  Words/paragraph: median={paras['word_count'].median():.0f}  p95={paras['word_count'].quantile(0.95):.0f}")
print(f"  Paragraphs/article: median={paras_per_article.median():.0f}  p95={paras_per_article.quantile(0.95):.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(paras["word_count"].clip(upper=300), bins=50, edgecolor="white", color=pal[1])
axes[0].axvline(paras["word_count"].median(), color="red", linestyle="--",
                label=f"Median {paras['word_count'].median():.0f}")
axes[0].set_xlabel("Words per paragraph  (clipped at 300)")
axes[0].set_ylabel("Paragraphs")
axes[0].set_title("Paragraph Word Count")
axes[0].legend()

axes[1].hist(paras_per_article, bins=40, edgecolor="white", color=pal[2])
axes[1].axvline(paras_per_article.median(), color="red", linestyle="--",
                label=f"Median {paras_per_article.median():.0f}")
axes[1].set_xlabel("Paragraphs per article")
axes[1].set_ylabel("Articles")
axes[1].set_title("Paragraphs per Article")
axes[1].legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "06_paragraph_distributions.png", dpi=150)
plt.close(fig)


# ── 7. Temporal coverage ──────────────────────────────────────────────────────

dated = success.dropna(subset=["publish_date"]).copy()
dated["month"] = dated["publish_date"].dt.to_period("M")
monthly = dated.groupby(["month", "pub_label"]).size().unstack(fill_value=0)

print(f"\n=== Temporal ===")
print(f"  Date range: {dated['publish_date'].min().date()} to {dated['publish_date'].max().date()}")
print(f"  Posts missing publish_date: {success['publish_date'].isna().sum()}")

fig, ax = plt.subplots(figsize=(14, 4))
monthly.plot(kind="area", ax=ax, stacked=True, alpha=0.75)
ax.set_xlabel("Month")
ax.set_ylabel("Posts")
ax.set_title("Temporal Coverage of Scraped Posts")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "07_temporal_coverage.png", dpi=150)
plt.close(fig)


print(f"\nFigures saved to {OUT_DIR}/")
