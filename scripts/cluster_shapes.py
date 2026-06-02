"""
Learned gradient shape labeling via k-means clustering.

Replaces the rule-based gradient_shape column with data-driven cluster assignments.
Cluster IDs are mapped to shape names by majority vote against the existing
rule-based labels. Small clusters (n < MIN_CLUSTER_SIZE) fall back to their
rule-based majority label rather than claiming a shape name, preventing a single
outlier article from polluting the shape vocabulary.

Steps:
  1. Load article_features.csv
  2. Select shape-relevant trajectory features
  3. Standardize and fit k-means for k=2..8; pick k by silhouette score
  4. Fit final model at chosen k (default: best by silhouette)
  5. Map cluster IDs to shape names; fall back to rule-based label for tiny clusters
  6. Print cross-tab and per-cluster feature profiles
  7. Overwrite gradient_shape in article_features.csv with learned labels

Usage:
    uv run scripts/cluster_shapes.py
    uv run scripts/cluster_shapes.py --k 4 --data-dir data/processed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Features that describe trajectory shape — deliberately excludes mean_complexity,
# engagement targets, word_count, and publication metadata so the clusters reflect
# structural shape rather than overall difficulty level or source identity.
SHAPE_FEATURES = [
    "complexity_slope",       # overall direction (ramp vs. resolution)
    "complexity_variance",    # how much complexity varies (plateau vs. rollercoaster)
    "peak_position",          # where the peak sits (cliff = early, ramp = late)
    "resolution_index",       # peak minus ending complexity (resolution shape)
    "jumpiness",              # variance of step-to-step changes (rollercoaster)
    "derivative_std",         # std of step-to-step changes (rollercoaster)
    "opening_complexity",     # relative opening difficulty (cliff)
    "ending_complexity",      # relative ending difficulty (resolution)
    "above_mean_rate",        # share of paragraphs above mean (plateau = ~0.5)
    "early_spike_score",      # max complexity in first 20% (cliff)
    "n_spikes",               # number of complexity spikes
]

RULE_LABEL_COL    = "gradient_shape"
LEARNED_LABEL_COL = "gradient_shape_learned"
RULE_SHAPES       = ["ramp", "cliff", "plateau", "rollercoaster", "resolution"]
MIN_CLUSTER_SIZE  = 5  # clusters smaller than this fall back to rule-based majority label


def load(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "article_features.csv"
    df = pd.read_csv(path)
    available = [f for f in SHAPE_FEATURES if f in df.columns]
    missing = set(SHAPE_FEATURES) - set(available)
    if missing:
        print(f"Warning: features not found in data, skipping: {sorted(missing)}")
    return df, available


def silhouette_sweep(X_scaled: np.ndarray, k_range: range) -> dict[int, float]:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = round(silhouette_score(X_scaled, labels), 4)
    return scores


def fit_kmeans(X_scaled: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    return km, labels


def map_clusters_to_shapes(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    k: int,
) -> dict[int, str]:
    """
    Assign each cluster ID a shape name by majority vote against the rule-based labels.
    Clusters below MIN_CLUSTER_SIZE skip the shape-name competition and fall back
    directly to their rule-based majority label (so a single outlier article doesn't
    claim a shape name). Larger clusters are assigned greedily by highest count.
    """
    mapping: dict[int, str] = {}
    claimed: set[str] = set()

    small = {cid for cid in range(k) if (cluster_labels == cid).sum() < MIN_CLUSTER_SIZE}

    # Greedy shape-name assignment for normal-sized clusters only
    candidates = []
    for cluster_id in range(k):
        if cluster_id in small:
            continue
        mask = cluster_labels == cluster_id
        counts = df.loc[mask, RULE_LABEL_COL].value_counts()
        for shape in counts.index:
            candidates.append((cluster_id, shape, counts[shape]))

    candidates.sort(key=lambda x: x[2], reverse=True)
    for cluster_id, shape, _ in candidates:
        if cluster_id not in mapping and shape not in claimed:
            mapping[cluster_id] = shape
            claimed.add(shape)

    # Small clusters: rule-based majority label (no shape name claimed)
    for cluster_id in small:
        mask = cluster_labels == cluster_id
        mapping[cluster_id] = df.loc[mask, RULE_LABEL_COL].mode()[0]
        print(f"  Cluster {cluster_id} (n={mask.sum()}) too small — falling back to rule-based label '{mapping[cluster_id]}'")

    # Any remaining unmapped clusters (shouldn't happen)
    for cluster_id in range(k):
        if cluster_id not in mapping:
            mapping[cluster_id] = f"cluster_{cluster_id}"

    return mapping


def print_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    mapping: dict[int, str],
    features: list[str],
    k: int,
) -> None:
    print("\n" + "=" * 70)
    print("Cluster feature profiles (means, standardized)")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[features].fillna(0)),
        columns=features,
        index=df.index,
    )
    X_scaled["_cluster"] = cluster_labels
    X_scaled["_rule_shape"] = df[RULE_LABEL_COL].values

    for cid in range(k):
        name = mapping.get(cid, f"cluster_{cid}")
        mask = cluster_labels == cid
        n = mask.sum()
        profile = X_scaled.loc[mask, features].mean().sort_values(ascending=False)
        rule_dist = df.loc[mask, RULE_LABEL_COL].value_counts()

        print(f"\nCluster {cid} -> '{name}'  (n={n})")
        print(f"  Rule-based breakdown: {rule_dist.to_dict()}")
        print("  Top features (mean z-score):")
        for feat, val in profile.head(4).items():
            bar = "|" * min(int(abs(val) * 6), 20)
            sign = "+" if val >= 0 else "-"
            print(f"    {feat:<28} {val:>+6.3f}  {bar}{sign}")


def main(data_dir: Path, k: int | None) -> None:
    df, features = load(data_dir)
    print(f"Loaded {len(df)} articles — using {len(features)} shape features.")

    X = df[features].fillna(df[features].median()).fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Silhouette sweep ──────────────────────────────────────────────────────
    print("\nSilhouette scores across k:")
    sweep = silhouette_sweep(X_scaled, range(2, 9))
    for kk, score in sweep.items():
        bar = "#" * int(score * 40)
        print(f"  k={kk}  {score:.4f}  {bar}")

    best_k = max(sweep, key=sweep.get)
    chosen_k = k if k is not None else best_k
    print(f"\nBest k by silhouette: {best_k}  |  Using k={chosen_k}")

    # ── Fit final model ───────────────────────────────────────────────────────
    km, cluster_labels = fit_kmeans(X_scaled, chosen_k)
    sil = silhouette_score(X_scaled, cluster_labels)
    print(f"Final model silhouette score: {sil:.4f}")

    mapping = map_clusters_to_shapes(df, cluster_labels, chosen_k)
    df[LEARNED_LABEL_COL] = pd.Series(cluster_labels).map(mapping).values

    # ── Cross-tab ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Cross-tab: rule-based '{RULE_LABEL_COL}' vs learned '{LEARNED_LABEL_COL}'")
    print("=" * 70)
    cross = pd.crosstab(
        df[RULE_LABEL_COL],
        df[LEARNED_LABEL_COL],
        margins=True,
    )
    print(cross.to_string())

    # Agreement rate (where both agree, ignoring numbering differences)
    agree = (df[RULE_LABEL_COL] == df[LEARNED_LABEL_COL]).sum()
    print(f"\nExact label agreement: {agree}/{len(df)} ({agree/len(df):.1%})")

    # ── Per-cluster profiles ──────────────────────────────────────────────────
    print_cluster_profiles(df, cluster_labels, mapping, features, chosen_k)

    # ── Overwrite gradient_shape with learned labels ──────────────────────────
    print("\nLearned label distribution:")
    print(df[LEARNED_LABEL_COL].value_counts().to_string())
    print("\nRule-based label distribution (for comparison):")
    print(df[RULE_LABEL_COL].value_counts().to_string())

    df = df.drop(columns=[RULE_LABEL_COL])
    df = df.rename(columns={LEARNED_LABEL_COL: RULE_LABEL_COL})

    out_path = data_dir / "article_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n'{RULE_LABEL_COL}' overwritten with learned labels in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Directory containing article_features.csv",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Number of clusters (default: chosen by silhouette score)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.k)
