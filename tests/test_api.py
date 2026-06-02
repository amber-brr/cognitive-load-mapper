import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import api.data as data_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    articles_df = pd.DataFrame([{
        "post_url": "https://example.com/a1",
        "title": "Test Article",
        "publication_name": "Test Pub",
        "source_key": "test_pub",
        "publication_url": "https://example.com",
        "publish_date": "2024-01-01",
        "word_count": 2000,
        "paragraph_count": 3,
        "like_count": 100.0,
        "comment_count": 20.0,
        "engagement_available": True,
        "n_paragraphs": 3,
        "mean_complexity": 0.0,
        "max_complexity": 1.0,
        "complexity_variance": 0.3,
        "complexity_range": 1.0,
        "complexity_slope": 0.05,
        "opening_complexity": -0.3,
        "middle_complexity": 0.0,
        "ending_complexity": 0.5,
        "peak_position": 0.8,
        "resolution_index": 0.5,
        "jumpiness": 0.1,
        "derivative_std": 0.3,
        "n_spikes": 1,
        "above_mean_rate": 0.5,
        "early_spike_score": 0.1,
        "mean_embed_dist": 0.3,
        "max_embed_dist": 0.6,
        "embed_dist_slope": 0.01,
        "mean_sentiment_compound": 0.1,
        "mean_sentiment_neg": 0.05,
        "negative_paragraph_rate": 0.1,
        "sentiment_variance": 0.05,
        "mean_abs_sentiment_shift": 0.1,
        "sentiment_shift_rate": 0.2,
        "sentiment_slope": 0.01,
        "gradient_shape": "plateau",
        "engagement_raw": 4.1,
        "engagement_z": 0.5,
        "above_median_engagement_by_publication": 1,
    }])
    para_df = pd.DataFrame([
        {"post_url": "https://example.com/a1", "paragraph_index": 0, "paragraph_text": "Para one.", "paragraph_position_norm": 0.0, "fk_grade": 8.0, "avg_sentence_length": 12.0, "complex_word_rate": 0.1, "type_token_ratio": 0.7, "ner_count": 2, "new_entity_count": 2, "embed_dist_prev": 0.0},
        {"post_url": "https://example.com/a1", "paragraph_index": 1, "paragraph_text": "Para two.", "paragraph_position_norm": 0.5, "fk_grade": 11.0, "avg_sentence_length": 18.0, "complex_word_rate": 0.2, "type_token_ratio": 0.6, "ner_count": 4, "new_entity_count": 3, "embed_dist_prev": 0.3},
        {"post_url": "https://example.com/a1", "paragraph_index": 2, "paragraph_text": "Para three.", "paragraph_position_norm": 1.0, "fk_grade": 14.0, "avg_sentence_length": 22.0, "complex_word_rate": 0.3, "type_token_ratio": 0.5, "ner_count": 5, "new_entity_count": 4, "embed_dist_prev": 0.5},
    ])
    articles_df.to_csv(tmp_path / "article_features.csv", index=False)
    para_df.to_csv(tmp_path / "para_features.csv", index=False)
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    data_module.load()
    monkeypatch.setattr(data_module, "load", lambda: None)  # prevent lifespan overwrite

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_list_articles(client):
    r = client.get("/articles")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["title"] == "Test Article"
    assert body[0]["gradient_shape"] == "plateau"


def test_list_articles_shape_filter(client):
    assert len(client.get("/articles?shape=plateau").json()) == 1
    assert len(client.get("/articles?shape=ramp").json()) == 0


def test_get_article(client):
    r = client.get("/articles/0")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Test Article"
    assert body["gradient_shape"] == "plateau"
    assert len(body["paragraphs"]) == 3
    assert "complexity_v1" in body["paragraphs"][0]


def test_get_article_not_found(client):
    assert client.get("/articles/999").status_code == 404


def test_rewrite_same_shape_returns_empty(client):
    r = client.post("/articles/0/rewrite", json={"target_shape": "plateau"})
    assert r.status_code == 200
    body = r.json()
    assert body["flagged_paragraphs"] == []
    assert body["message"] is not None


def test_rewrite_different_shape(client):
    with patch("api.rewrite.rewrite_paragraph", return_value="Simpler version."):
        r = client.post("/articles/0/rewrite", json={"target_shape": "resolution"})
    assert r.status_code == 200
    body = r.json()
    assert body["current_shape"] == "plateau"
    assert body["target_shape"] == "resolution"
    assert len(body["flagged_paragraphs"]) > 0
    for fp in body["flagged_paragraphs"]:
        assert fp["rewritten_text"] == "Simpler version."


def test_rewrite_invalid_shape(client):
    assert client.post("/articles/0/rewrite", json={"target_shape": "invalid"}).status_code == 422
