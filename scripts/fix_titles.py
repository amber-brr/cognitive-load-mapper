"""One-time script: re-extract og:title from raw HTML and patch posts_scraped.csv + article_features.csv."""
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def extract_og_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("meta", property="og:title")
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None


def main():
    scraped = pd.read_csv("data/processed/posts_scraped.csv")
    fixed = 0

    for i, row in scraped.iterrows():
        path = row.get("raw_html_path")
        if not path:
            continue
        html_path = Path(path)
        if not html_path.exists():
            continue
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        og_title = extract_og_title(html)
        if og_title and og_title != row["title"]:
            scraped.at[i, "title"] = og_title
            fixed += 1

    print(f"Fixed {fixed} titles in posts_scraped.csv")
    scraped.to_csv("data/processed/posts_scraped.csv", index=False)

    # Patch article_features.csv using post_url as the join key
    features = pd.read_csv("data/processed/article_features.csv")
    title_map = scraped.set_index("post_url")["title"].to_dict()
    before = features["title"].copy()
    features["title"] = features["post_url"].map(title_map).fillna(features["title"])
    changed = (features["title"] != before).sum()
    print(f"Fixed {changed} titles in article_features.csv")
    features.to_csv("data/processed/article_features.csv", index=False)


if __name__ == "__main__":
    main()
