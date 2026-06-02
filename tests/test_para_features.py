"""
Tests for para_features._lexical_features VADER sentiment scores.
Note: importing para_features loads spacy + SentenceTransformer (~10-30s first run).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from para_features import _lexical_features


def test_lexical_features_includes_all_vader_keys():
    result = _lexical_features("This is a wonderful day. I love the sunshine.")
    assert "vader_compound" in result
    assert "vader_pos" in result
    assert "vader_neg" in result
    assert "vader_neu" in result


def test_vader_compound_positive_text():
    result = _lexical_features("I love this amazing, wonderful, fantastic day!")
    assert result["vader_compound"] > 0


def test_vader_compound_negative_text():
    result = _lexical_features("This is terrible, awful, and completely disgusting.")
    assert result["vader_compound"] < 0


def test_vader_component_scores_sum_to_one():
    result = _lexical_features("Some neutral text about nothing in particular.")
    total = result["vader_pos"] + result["vader_neg"] + result["vader_neu"]
    assert abs(total - 1.0) < 0.01


def test_vader_empty_string_returns_zeros():
    result = _lexical_features("")
    assert result["vader_compound"] == 0.0
    assert result["vader_pos"] == 0.0
    assert result["vader_neg"] == 0.0
    assert result["vader_neu"] == 0.0
