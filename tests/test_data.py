import pandas as pd
import pytest
from pathlib import Path

import api.data as data_module
from api.data import flag_paragraphs, get_article, list_articles


@pytest.fixture(autouse=True)
def load_test_data(tmp_path: Path, monkeypatch):
    articles_df = pd.DataFrame([
        {
            "post_url": "https://example.com/article-1",
            "title": "Deep Dive Into Systems",
            "publication_name": "Construction Physics",
            "source_key": "construction_physics",
            "publication_url": "https://example.com",
            "publish_date": "2024-01-15",
            "word_count": 3200,
            "paragraph_count": 20,
            "like_count": 150.0,
            "comment_count": 30.0,
            "engagement_available": True,
            "n_paragraphs": 3,
            "mean_complexity": 0.2,
            "max_complexity": 1.5,
            "complexity_variance": 0.4,
            "complexity_range": 1.5,
            "complexity_slope": 0.05,
            "opening_complexity": -0.3,
            "middle_complexity": 0.2,
            "ending_complexity": 0.7,
            "peak_position": 0.8,
            "resolution_index": 0.8,
            "jumpiness": 0.1,
            "derivative_std": 0.3,
            "n_spikes": 3,
            "above_mean_rate": 0.5,
            "early_spike_score": 0.1,
            "mean_embed_dist": 0.3,
            "max_embed_dist": 0.7,
            "embed_dist_slope": 0.01,
            "mean_sentiment_compound": 0.1,
            "mean_sentiment_neg": 0.05,
            "negative_paragraph_rate": 0.1,
            "sentiment_variance": 0.05,
            "mean_abs_sentiment_shift": 0.1,
            "sentiment_shift_rate": 0.2,
            "sentiment_slope": 0.01,
            "gradient_shape": "ramp",
            "engagement_raw": 4.9,
            "engagement_z": 1.2,
            "above_median_engagement_by_publication": 1,
        },
        {
            "post_url": "https://example.com/article-2",
            "title": "The Plateau Essay",
            "publication_name": "Experimental History",
            "source_key": "experimental_history",
            "publication_url": "https://experimental.com",
            "publish_date": "2024-02-10",
            "word_count": 1800,
            "paragraph_count": 10,
            "like_count": 50.0,
            "comment_count": 10.0,
            "engagement_available": True,
            "n_paragraphs": 2,
            "mean_complexity": -0.1,
            "max_complexity": 0.3,
            "complexity_variance": 0.05,
            "complexity_range": 0.4,
            "complexity_slope": 0.01,
            "opening_complexity": -0.1,
            "middle_complexity": -0.1,
            "ending_complexity": -0.1,
            "peak_position": 0.5,
            "resolution_index": 0.2,
            "jumpiness": 0.02,
            "derivative_std": 0.1,
            "n_spikes": 0,
            "above_mean_rate": 0.5,
            "early_spike_score": 0.0,
            "mean_embed_dist": 0.2,
            "max_embed_dist": 0.4,
            "embed_dist_slope": 0.0,
            "mean_sentiment_compound": 0.05,
            "mean_sentiment_neg": 0.02,
            "negative_paragraph_rate": 0.05,
            "sentiment_variance": 0.02,
            "mean_abs_sentiment_shift": 0.05,
            "sentiment_shift_rate": 0.1,
            "sentiment_slope": 0.0,
            "gradient_shape": "plateau",
            "engagement_raw": 3.6,
            "engagement_z": -0.3,
            "above_median_engagement_by_publication": 0,
        },
    ])
    para_df = pd.DataFrame([
        {"post_url": "https://example.com/article-1", "paragraph_index": 0, "paragraph_text": "Simple intro.", "paragraph_position_norm": 0.0, "fk_grade": 8.0, "avg_sentence_length": 12.0, "complex_word_rate": 0.1, "type_token_ratio": 0.7, "ner_count": 2, "new_entity_count": 2, "embed_dist_prev": 0.0},
        {"post_url": "https://example.com/article-1", "paragraph_index": 1, "paragraph_text": "Middle complexity.", "paragraph_position_norm": 0.5, "fk_grade": 11.0, "avg_sentence_length": 18.0, "complex_word_rate": 0.2, "type_token_ratio": 0.6, "ner_count": 4, "new_entity_count": 3, "embed_dist_prev": 0.3},
        {"post_url": "https://example.com/article-1", "paragraph_index": 2, "paragraph_text": "Dense complex ending.", "paragraph_position_norm": 1.0, "fk_grade": 15.0, "avg_sentence_length": 25.0, "complex_word_rate": 0.35, "type_token_ratio": 0.5, "ner_count": 6, "new_entity_count": 5, "embed_dist_prev": 0.5},
        {"post_url": "https://example.com/article-2", "paragraph_index": 0, "paragraph_text": "Steady para one.", "paragraph_position_norm": 0.0, "fk_grade": 10.0, "avg_sentence_length": 15.0, "complex_word_rate": 0.15, "type_token_ratio": 0.65, "ner_count": 3, "new_entity_count": 3, "embed_dist_prev": 0.0},
        {"post_url": "https://example.com/article-2", "paragraph_index": 1, "paragraph_text": "Steady para two.", "paragraph_position_norm": 1.0, "fk_grade": 10.5, "avg_sentence_length": 15.5, "complex_word_rate": 0.16, "type_token_ratio": 0.64, "ner_count": 3, "new_entity_count": 2, "embed_dist_prev": 0.2},
    ])
    articles_df.to_csv(tmp_path / "article_features.csv", index=False)
    para_df.to_csv(tmp_path / "para_features.csv", index=False)
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    data_module.load()


def test_list_articles_returns_all():
    assert len(list_articles(None, None, 20, 0)) == 2


def test_list_articles_filter_publication():
    result = list_articles("Construction", None, 20, 0)
    assert len(result) == 1
    assert result[0]["publication_name"] == "Construction Physics"


def test_list_articles_filter_shape():
    result = list_articles(None, "plateau", 20, 0)
    assert len(result) == 1
    assert result[0]["gradient_shape"] == "plateau"


def test_list_articles_pagination():
    assert len(list_articles(None, None, 1, 1)) == 1


def test_get_article_valid():
    row, paras = get_article(0)
    assert row is not None
    assert row["title"] == "Deep Dive Into Systems"
    assert len(paras) == 3
    assert all("complexity_v1" in p for p in paras)


def test_get_article_invalid():
    row, paras = get_article(999)
    assert row is None
    assert paras == []


def test_flag_paragraphs_ramp_flags_early_complex_and_late_simple():
    paras = [
        {"paragraph_index": 0, "paragraph_text": "Complex early.", "complexity_v1": 1.0, "paragraph_position_norm": 0.1},
        {"paragraph_index": 1, "paragraph_text": "Medium.", "complexity_v1": 0.0, "paragraph_position_norm": 0.5},
        {"paragraph_index": 2, "paragraph_text": "Simple late.", "complexity_v1": -0.5, "paragraph_position_norm": 0.9},
    ]
    flagged = flag_paragraphs(paras, "ramp")
    indices = [f["paragraph_index"] for f in flagged]
    assert 0 in indices  # early para too complex for ramp
    assert 2 in indices  # late para too simple for ramp


def test_flag_paragraphs_resolution_flags_late_complex():
    paras = [
        {"paragraph_index": 0, "paragraph_text": "Intro.", "complexity_v1": 0.0, "paragraph_position_norm": 0.0},
        {"paragraph_index": 1, "paragraph_text": "High complexity at end.", "complexity_v1": 1.5, "paragraph_position_norm": 0.9},
    ]
    flagged = flag_paragraphs(paras, "resolution")
    assert any(f["paragraph_index"] == 1 for f in flagged)


def test_flag_paragraphs_capped_at_three():
    # 6 early high-complexity paras + 4 late low-complexity: all 10 should be flagged for ramp,
    # but the cap returns only 3 sorted by largest deviation from mean.
    paras = [
        {"paragraph_index": i, "paragraph_text": f"Para {i}.", "complexity_v1": 2.0, "paragraph_position_norm": 0.1}
        for i in range(6)
    ] + [
        {"paragraph_index": i + 6, "paragraph_text": f"Para {i + 6}.", "complexity_v1": -1.0, "paragraph_position_norm": 0.9}
        for i in range(4)
    ]
    flagged = flag_paragraphs(paras, "ramp")
    assert len(flagged) == 3


def test_flag_paragraphs_cliff_flags_non_opening_high_complexity():
    paras = [
        {"paragraph_index": 0, "paragraph_text": "Low.", "complexity_v1": 0.0, "paragraph_position_norm": 0.0},
        {"paragraph_index": 1, "paragraph_text": "Low.", "complexity_v1": 0.0, "paragraph_position_norm": 0.3},
        {"paragraph_index": 2, "paragraph_text": "Very complex late.", "complexity_v1": 3.0, "paragraph_position_norm": 0.8},
    ]
    # mean=1.0, std=1.41, threshold=1.0+0.5*1.41=1.71
    # Para 2: pos=0.8 > 0.2, c=3.0 > 1.71 → flagged
    flagged = flag_paragraphs(paras, "cliff")
    assert any(f["paragraph_index"] == 2 for f in flagged)


def test_flag_paragraphs_plateau_flags_outliers():
    paras = [
        {"paragraph_index": 0, "paragraph_text": "Normal.", "complexity_v1": 0.0, "paragraph_position_norm": 0.0},
        {"paragraph_index": 1, "paragraph_text": "Normal.", "complexity_v1": 0.1, "paragraph_position_norm": 0.5},
        {"paragraph_index": 2, "paragraph_text": "Very complex outlier.", "complexity_v1": 3.0, "paragraph_position_norm": 1.0},
    ]
    # mean≈1.03, std≈1.37; para 2: |3.0-1.03|=1.97 > 1.37 → flagged
    flagged = flag_paragraphs(paras, "plateau")
    assert any(f["paragraph_index"] == 2 for f in flagged)


def test_flag_paragraphs_rollercoaster_flags_flat_runs():
    paras = [
        {"paragraph_index": 0, "paragraph_text": "High.", "complexity_v1": 2.0, "paragraph_position_norm": 0.0},
        {"paragraph_index": 1, "paragraph_text": "High.", "complexity_v1": 2.0, "paragraph_position_norm": 0.25},
        {"paragraph_index": 2, "paragraph_text": "High.", "complexity_v1": 2.0, "paragraph_position_norm": 0.5},
        {"paragraph_index": 3, "paragraph_text": "Low.", "complexity_v1": -1.0, "paragraph_position_norm": 0.75},
        {"paragraph_index": 4, "paragraph_text": "Low.", "complexity_v1": -1.0, "paragraph_position_norm": 1.0},
    ]
    # mean=1.0, std≈1.26; paras 0,1,2 all have c=2.0 > mean+0.1*std≈1.13 → sign=1
    # Three consecutive +1 signs at indices 0,1,2 → para 1 (index 1) flagged
    flagged = flag_paragraphs(paras, "rollercoaster")
    assert any(f["paragraph_index"] == 1 for f in flagged)
