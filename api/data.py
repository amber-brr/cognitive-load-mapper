import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))

COMPLEXITY_FEATURES = [
    "fk_grade", "avg_sentence_length", "complex_word_rate",
    "type_token_ratio", "ner_count", "new_entity_count", "embed_dist_prev",
]

articles: pd.DataFrame = pd.DataFrame()
paragraphs: pd.DataFrame = pd.DataFrame()


def load() -> None:
    global articles, paragraphs
    articles = (
        pd.read_csv(DATA_DIR / "article_features.csv")
        .reset_index(drop=True)
        .rename_axis("article_id")
        .reset_index()
    )
    paragraphs = pd.read_csv(DATA_DIR / "para_features.csv")

    available = [f for f in COMPLEXITY_FEATURES if f in paragraphs.columns]
    X = paragraphs[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=available, index=paragraphs.index
    )
    paragraphs["complexity_v1"] = X_scaled.mean(axis=1)
    logger.info("Loaded %d articles, %d paragraphs.", len(articles), len(paragraphs))


def _clean(row: dict) -> dict:
    result = {}
    for k, v in row.items():
        try:
            result[k] = None if pd.isna(v) else v
        except (TypeError, ValueError):
            result[k] = v
    return result


def list_articles(
    publication: str | None,
    shape: str | None,
    limit: int,
    offset: int,
) -> list[dict]:
    df = articles.copy()
    if publication:
        df = df[df["publication_name"].str.contains(publication, case=False, na=False)]
    if shape:
        df = df[df["gradient_shape"] == shape]
    df = df.iloc[offset : offset + limit]
    cols = ["article_id", "title", "publication_name", "gradient_shape",
            "word_count", "mean_complexity", "engagement_z"]
    subset = df[[c for c in cols if c in df.columns]]
    return [_clean(row) for row in subset.to_dict(orient="records")]


def get_article(article_id: int) -> tuple[dict | None, list[dict]]:
    if articles.empty or article_id < 0 or article_id >= len(articles):
        return None, []
    row = _clean(articles.iloc[article_id].to_dict())
    paras = (
        paragraphs[paragraphs["post_url"] == row["post_url"]]
        .sort_values("paragraph_index")
    )
    cols = ["paragraph_index", "paragraph_text", "complexity_v1", "paragraph_position_norm"]
    return row, paras[[c for c in cols if c in paras.columns]].to_dict(orient="records")


def flag_paragraphs(
    para_rows: list[dict],
    target_shape: str,
) -> list[dict]:
    if not para_rows:
        return []

    scores = np.array([p["complexity_v1"] for p in para_rows])
    mean_s = float(scores.mean())
    std_s = float(scores.std()) or 1.0
    n = len(scores)
    flagged: list[dict] = []

    if target_shape == "ramp":
        for p in para_rows:
            pos, c = p["paragraph_position_norm"], p["complexity_v1"]
            if pos < 0.4 and c > mean_s:
                flagged.append({**p, "reason": f"Too complex at position {pos:.2f} — early paragraphs should build gradually toward a ramp."})
            elif pos > 0.6 and c < mean_s:
                flagged.append({**p, "reason": f"Too simple at position {pos:.2f} — late paragraphs should carry the most complexity in a ramp."})

    elif target_shape == "resolution":
        for p in para_rows:
            pos, c = p["paragraph_position_norm"], p["complexity_v1"]
            if pos > 0.7 and c > mean_s:
                flagged.append({**p, "reason": f"Too complex at position {pos:.2f} — a resolution shape eases off toward the end."})

    elif target_shape == "cliff":
        for p in para_rows:
            pos, c = p["paragraph_position_norm"], p["complexity_v1"]
            if pos > 0.2 and c > mean_s + 0.5 * std_s:
                flagged.append({**p, "reason": f"High complexity at position {pos:.2f} — a cliff concentrates difficulty in the opening only."})

    elif target_shape == "plateau":
        for p in para_rows:
            c = p["complexity_v1"]
            if abs(c - mean_s) > std_s:
                word = "complex" if c > mean_s else "simple"
                flagged.append({**p, "reason": f"Too {word} relative to the article mean — a plateau needs consistent complexity throughout."})

    elif target_shape == "rollercoaster":
        signs = [
            1 if p["complexity_v1"] > mean_s + 0.1 * std_s
            else (-1 if p["complexity_v1"] < mean_s - 0.1 * std_s else 0)
            for p in para_rows
        ]
        for i in range(1, n - 1):
            if signs[i] != 0 and signs[i - 1] == signs[i] == signs[i + 1]:
                p = para_rows[i]
                word = "above" if signs[i] > 0 else "below"
                flagged.append({**p, "reason": f"Three consecutive paragraphs {word} average complexity — a rollercoaster needs more alternation here."})

    flagged.sort(key=lambda p: abs(p["complexity_v1"] - mean_s), reverse=True)
    return flagged[:3]
