"""
Per-paragraph feature extraction for the Cognitive Complexity Gradient Mapper.
Input:  data/processed/paragraphs.csv
Output: data/processed/para_features.csv
Run:    python para_features.py

Requires:
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt_tab')"
"""

import re
import warnings
from pathlib import Path

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import spacy
import textstat
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
nltk.download("punkt_tab", quiet=True)
nltk.download("vader_lexicon", quiet=True)

IN_PATH  = Path("data/processed/paragraphs.csv")
OUT_PATH = Path("data/processed/para_features.csv")

_nlp      = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_vader = SentimentIntensityAnalyzer()

_TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "additionally",
    "consequently", "nevertheless", "nonetheless", "similarly", "likewise",
    "although", "whereas", "meanwhile", "thus", "hence", "accordingly",
    "subsequently", "conversely", "alternatively",
}
_EXAMPLE_RE = re.compile(
    r"\bfor (example|instance)\b|\be\.g\.?\b|\bsuch as\b|\bnamely\b", re.I
)
_TRANSITION_RE = re.compile(r"\b(" + "|".join(_TRANSITION_WORDS) + r")\b", re.I)


# ── Per-paragraph feature helpers ─────────────────────────────────────────────

def _alpha_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z']+\b", text)


def _mattr(words: list[str], window: int = 50) -> float:
    """Moving-average TTR: more stable than raw TTR for short texts."""
    n = len(words)
    if n < window:
        return len({w.lower() for w in words}) / max(n, 1)
    ttrs = [
        len({words[i + j].lower() for j in range(window)}) / window
        for i in range(n - window + 1)
    ]
    return float(np.mean(ttrs))


def _sents(text: str) -> list[str]:
    try:
        return sent_tokenize(text)
    except Exception:
        return re.split(r"(?<=[.!?])\s+", text)


def _lexical_features(text: str) -> dict:
    words    = _alpha_words(text)
    n_words  = max(len(words), 1)
    n_total  = max(len(text.split()), 1)  # total tokens incl. numbers/punct
    sents    = _sents(text)
    n_sents  = max(len(sents), 1)
    n_lex    = textstat.lexicon_count(text, removepunct=True)
    vs       = _vader.polarity_scores(text)

    return {
        "sentence_count":      n_sents,
        "avg_sentence_length": n_words / n_sents,
        "avg_word_length":     float(np.mean([len(w) for w in words])) if words else 0.0,
        # NaN for very short paragraphs — 0.0 would falsely read as kindergarten level
        "fk_grade":            textstat.flesch_kincaid_grade(text) if n_lex >= 10 else float("nan"),
        "gunning_fog":         textstat.gunning_fog(text)          if n_lex >= 10 else float("nan"),
        "complex_word_rate":   textstat.polysyllabcount(text) / n_words,
        # MATTR is more stable than raw TTR for short paragraphs
        "type_token_ratio":    _mattr(words),
        # number density over total tokens, not alpha-only words
        "number_density":      len(re.findall(r"\b\d[\d,.]*\b", text)) / n_total,
        "transition_flag":     int(bool(_TRANSITION_RE.search(text))),
        "example_flag":        int(bool(_EXAMPLE_RE.search(text))),
        "question_count":      sum(1 for s in sents if s.strip().endswith("?")),
        "vader_compound":      vs["compound"],
        "vader_pos":           vs["pos"],
        "vader_neg":           vs["neg"],
        "vader_neu":           vs["neu"],
    }


# ── Per-article processing ─────────────────────────────────────────────────────

def _process_article(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("paragraph_index").reset_index(drop=True)
    texts = [str(t) if pd.notna(t) else "" for t in group["paragraph_text"]]

    feat_rows = [_lexical_features(t) for t in texts]

    seen: set[str] = set()
    for i, doc in enumerate(_nlp.pipe(texts, batch_size=32)):
        ents = {e.text.lower() for e in doc.ents}
        new  = ents - seen
        feat_rows[i]["ner_count"]        = len(doc.ents)
        feat_rows[i]["new_entity_count"] = len(new)
        feat_rows[i]["new_entity_rate"]  = len(new) / max(len(ents), 1)
        seen.update(ents)

    if len(texts) >= 2:
        embs  = _embedder.encode(texts, batch_size=64, show_progress_bar=False)
        dists = [float("nan")] + list(1.0 - cosine_similarity(embs[:-1], embs[1:]).diagonal())
    else:
        dists = [float("nan")] * len(texts)

    n         = len(texts)
    threshold = max(1, round(n * 0.25))
    positions = np.arange(n)
    feat_df   = pd.DataFrame(feat_rows)
    feat_df["embed_dist_prev"]         = dists
    feat_df["paragraph_position_norm"] = positions / max(n - 1, 1)
    feat_df["is_intro_paragraph"]      = (positions < threshold).astype(int)
    feat_df["is_ending_paragraph"]     = (positions >= n - threshold).astype(int)

    index_cols = group[["post_url", "paragraph_index", "paragraph_text", "word_count"]].reset_index(drop=True)
    return pd.concat([index_cols, feat_df], axis=1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    paras = pd.read_csv(IN_PATH)
    n_articles = paras["post_url"].nunique()
    print(f"Input:  {len(paras)} paragraphs across {n_articles} articles")

    parts = []
    for _, group in tqdm(
        paras.groupby("post_url", sort=False),
        total=n_articles,
        desc="para_features",
    ):
        parts.append(_process_article(group))

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Output: {len(out)} rows × {out.shape[1]} columns → {OUT_PATH}")


if __name__ == "__main__":
    main()
