"""
Engagement prediction models for the Cognitive Complexity Gradient Mapper.

Compares feature sets against two targets:
  - Regression:      engagement_z (within-publication normalized log-engagement)
  - Classification:  above_median_engagement_by_publication (binary)

Feature sets:
  baseline_fk           mean Flesch-Kincaid grade (readability-only baseline)
  baseline_mean         mean_complexity only
  gradient              all article-level gradient features (no source controls)
  gradient_ctrl         gradient features + word_count + source_key dummies
  gradient_no_embed     gradient minus embedding-distance features
  gradient_no_sentiment gradient minus VADER sentiment features
  gradient_no_shape     gradient minus trajectory/shape features
  readability_only      FK grade + opening/middle/ending complexity

Cross-validation uses GroupKFold(n_splits=5) with groups=source_key so that
each fold holds out a full publication — preventing the model from memorising
publication-level baselines.

Usage:
    uv run scripts/train.py
    uv run scripts/train.py --data-dir data/processed --output outputs/model_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

_EMBED_FEATURES = [
    "mean_embed_dist",
    "max_embed_dist",
    "embed_dist_slope",
]

_SENTIMENT_FEATURES = [
    "mean_sentiment_compound",
    "mean_sentiment_neg",
    "negative_paragraph_rate",
    "sentiment_variance",
    "mean_abs_sentiment_shift",
    "sentiment_shift_rate",
    "sentiment_slope",
]

_SHAPE_FEATURES = [
    "peak_position",
    "resolution_index",
    "jumpiness",
    "derivative_std",
]

_READABILITY_FEATURES = [
    "mean_fk_grade",
    "opening_complexity",
    "middle_complexity",
    "ending_complexity",
]

GRADIENT_FEATURES = [
    "mean_complexity",
    "max_complexity",
    "complexity_variance",
    "complexity_range",
    "complexity_slope",
    "opening_complexity",
    "middle_complexity",
    "ending_complexity",
    "peak_position",
    "resolution_index",
    "jumpiness",
    "derivative_std",
    "n_spikes",
    "above_mean_rate",
    "early_spike_score",
    *_EMBED_FEATURES,
    *_SENTIMENT_FEATURES,
]

REGRESSION_TARGET = "engagement_z"
CLASSIFICATION_TARGET = "above_median_engagement_by_publication"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> pd.DataFrame:
    af = pd.read_csv(data_dir / "article_features.csv")
    pf = pd.read_csv(data_dir / "para_features.csv")

    mean_fk = pf.groupby("post_url")["fk_grade"].mean().rename("mean_fk_grade")
    af = af.merge(mean_fk, on="post_url", how="left")

    af = af.dropna(subset=[REGRESSION_TARGET, CLASSIFICATION_TARGET, "source_key"])

    # Drop rows missing all gradient features (shouldn't happen but be safe)
    af = af.dropna(subset=["mean_complexity"])

    print(f"Articles loaded: {len(af)}")
    print(f"Source distribution:\n{af['source_key'].value_counts().to_string()}\n")
    return af


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    medians = df[feature_cols].median()
    return df[feature_cols].fillna(medians).fillna(0).values


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":     round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":    round(root_mean_squared_error(y_true, y_pred), 4),
        "Spearman": round(float(spearmanr(y_true, y_pred).statistic), 4),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "F1":      round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Cross-validated training
# ---------------------------------------------------------------------------

def cv_evaluate(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_clf: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Run GroupKFold CV; return averaged regression and classification metrics."""
    gkf = GroupKFold(n_splits=n_splits)

    reg_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    clf_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])

    reg_metrics: list[dict] = []
    clf_metrics: list[dict] = []

    for train_idx, test_idx in gkf.split(X, y_reg, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_reg_tr, y_reg_te = y_reg[train_idx], y_reg[test_idx]
        y_clf_tr, y_clf_te = y_clf[train_idx], y_clf[test_idx]

        reg_pipe.fit(X_tr, y_reg_tr)
        y_reg_pred = reg_pipe.predict(X_te)
        reg_metrics.append(evaluate_regression(y_reg_te, y_reg_pred))

        # skip fold if only one class present
        if len(np.unique(y_clf_tr)) < 2:
            continue
        clf_pipe.fit(X_tr, y_clf_tr)
        y_clf_pred = clf_pipe.predict(X_te)
        y_clf_prob = clf_pipe.predict_proba(X_te)[:, 1]
        clf_metrics.append(evaluate_classification(y_clf_te, y_clf_pred, y_clf_prob))

    avg_reg = {k: round(np.mean([m[k] for m in reg_metrics]), 4) for k in reg_metrics[0]}
    avg_clf = {k: round(np.mean([m[k] for m in clf_metrics]), 4) for k in clf_metrics[0]}
    return {**avg_reg, **avg_clf}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, output_path: Path) -> None:
    df = load_data(data_dir)

    groups = df["source_key"].values
    y_reg = df[REGRESSION_TARGET].values.astype(float)
    y_clf = df[CLASSIFICATION_TARGET].values.astype(int)

    # One-hot encode source_key for the controlled model
    source_dummies = pd.get_dummies(df["source_key"], drop_first=True).astype(float)

    _grad_no_embed     = [f for f in GRADIENT_FEATURES if f not in _EMBED_FEATURES]
    _grad_no_sentiment = [f for f in GRADIENT_FEATURES if f not in _SENTIMENT_FEATURES]
    _grad_no_shape     = [f for f in GRADIENT_FEATURES if f not in _SHAPE_FEATURES]

    feature_sets: dict[str, list[str]] = {
        "baseline_fk":           ["mean_fk_grade"],
        "baseline_mean":         ["mean_complexity"],
        "gradient":              GRADIENT_FEATURES,
        "gradient_no_embed":     _grad_no_embed,
        "gradient_no_sentiment": _grad_no_sentiment,
        "gradient_no_shape":     _grad_no_shape,
        "readability_only":      _READABILITY_FEATURES,
    }

    results = []

    for name, cols in feature_sets.items():
        available = [c for c in cols if c in df.columns]
        X = build_feature_matrix(df, available)
        metrics = cv_evaluate(X, y_reg, y_clf, groups)
        results.append({"model": name, "n_features": len(available), **metrics})
        print(f"[{name}]  {metrics}")

    # Gradient + controls (source dummies + word_count)
    X_grad = build_feature_matrix(df, [c for c in GRADIENT_FEATURES if c in df.columns])
    X_ctrl = np.hstack([
        X_grad,
        df[["word_count"]].fillna(df["word_count"].median()).values,
        source_dummies.values,
    ])
    metrics = cv_evaluate(X_ctrl, y_reg, y_clf, groups)
    results.append({"model": "gradient_ctrl", "n_features": X_ctrl.shape[1], **metrics})
    print(f"[gradient_ctrl]  {metrics}")

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # --- Feature importance: fit on full data, read standardised coefficients ---
    grad_cols = [c for c in GRADIENT_FEATURES if c in df.columns]
    ctrl_cols = grad_cols + ["word_count"] + list(source_dummies.columns)

    print("\n" + "=" * 60)
    print("Feature importance (full-data fit, standardised coefficients)")
    print("=" * 60)

    interpretation_specs = [
        ("gradient        -> engagement_z",    grad_cols, X_grad, y_reg,  "regression"),
        ("gradient        -> above_median",   grad_cols, X_grad, y_clf,  "classification"),
        ("gradient_ctrl   -> engagement_z",   ctrl_cols, X_ctrl, y_reg,  "regression"),
        ("gradient_ctrl   -> above_median",   ctrl_cols, X_ctrl, y_clf,  "classification"),
    ]

    for title, feature_names, X, y, task in interpretation_specs:
        if task == "regression":
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            pipe.fit(X, y)
            coefs = pipe.named_steps["model"].coef_
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
            pipe.fit(X, y)
            coefs = pipe.named_steps["model"].coef_[0]

        ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n{title}")
        print(f"  {'feature':<35} {'coef':>8}")
        for feat, coef in ranked[:10]:
            bar = "+" if coef > 0 else "-"
            print(f"  {feat:<35} {coef:>+8.4f}  {'|' * min(int(abs(coef) * 10), 30)}{bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model_results.csv"))
    args = parser.parse_args()
    main(args.data_dir, args.output)
