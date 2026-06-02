"""
Tests for article_features._sentiment_gradient.
Importing article_features is fast (no heavy model loading at module level).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from article_features import _sentiment_gradient


def _group(compounds: list[float], negs: list[float] | None = None) -> pd.DataFrame:
    """Helper: build a minimal paragraph group DataFrame."""
    n = len(compounds)
    return pd.DataFrame({
        "paragraph_index": list(range(n)),
        "vader_compound":  compounds,
        "vader_neg":       negs if negs is not None else [0.0] * n,
    })


def test_sentiment_gradient_returns_all_expected_keys():
    group = _group([0.5, -0.3, 0.2, -0.1, 0.4])
    result = _sentiment_gradient(group)
    expected_keys = {
        "mean_sentiment_compound",
        "mean_sentiment_neg",
        "negative_paragraph_rate",
        "sentiment_variance",
        "mean_abs_sentiment_shift",
        "sentiment_shift_rate",
        "sentiment_slope",
    }
    assert set(result.keys()) == expected_keys


def test_sentiment_gradient_returns_empty_dict_when_columns_absent():
    group = pd.DataFrame({"paragraph_index": [0, 1], "unrelated_col": [1.0, 2.0]})
    assert _sentiment_gradient(group) == {}


def test_negative_paragraph_rate_counts_below_threshold():
    # compounds: 0.5, -0.1, -0.3, 0.2, -0.06 → 3 below -0.05 → rate = 0.6
    group = _group([0.5, -0.1, -0.3, 0.2, -0.06])
    result = _sentiment_gradient(group)
    assert abs(result["negative_paragraph_rate"] - 0.6) < 0.01


def test_sentiment_shift_rate_alternating_polarity():
    # +, -, +, - → every adjacent pair is a clear polarity flip → rate = 1.0
    group = _group([0.7, -0.7, 0.7, -0.7])
    result = _sentiment_gradient(group)
    assert result["sentiment_shift_rate"] == 1.0


def test_sentiment_shift_rate_all_positive_no_shifts():
    # all clearly positive → no polarity flips → rate = 0.0
    group = _group([0.6, 0.5, 0.7, 0.4])
    result = _sentiment_gradient(group)
    assert result["sentiment_shift_rate"] == 0.0


def test_sentiment_slope_negative_for_declining_sentiment():
    group = _group([0.8, 0.4, 0.0, -0.4, -0.8])
    result = _sentiment_gradient(group)
    assert result["sentiment_slope"] < 0


def test_sentiment_slope_positive_for_improving_sentiment():
    group = _group([-0.8, -0.4, 0.0, 0.4, 0.8])
    result = _sentiment_gradient(group)
    assert result["sentiment_slope"] > 0


def test_single_paragraph_no_errors():
    group = _group([0.5], negs=[0.0])
    result = _sentiment_gradient(group)
    assert result["sentiment_shift_rate"] == 0.0
    assert result["mean_abs_sentiment_shift"] == 0.0
    assert result["sentiment_slope"] == 0.0


def test_mean_sentiment_neg_uses_neg_column():
    group = _group([0.0, 0.0], negs=[0.2, 0.4])
    result = _sentiment_gradient(group)
    assert abs(result["mean_sentiment_neg"] - 0.3) < 0.001
