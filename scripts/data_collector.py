"""
Cognitive Complexity Gradient Mapper — Data Collection
Scrapes public Substack publications for article text and engagement metrics.
uv run run_data_collection_revised.py --output-dir data/processed --max-posts-per-publication 200 --log-level DEBUG
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "UCLA Stat418 Student (academic research; contact: amberjiang@g.ucla.edu)"
SLEEP_MIN = 2.0
SLEEP_MAX = 3.0
REQUEST_TIMEOUT = 30
ENGAGEMENT_TEST_N = 5
MIN_WORD_COUNT = 800
MIN_PARAGRAPH_COUNT = 6
MIN_PARAGRAPH_WORDS = 10

SEED_PUBLICATIONS = [
    {
        "publication_url": "https://www.construction-physics.com",
        "source_key": "construction_physics",
        "discovery_source": "handpicked_construction_physics",
    },
    {
        "publication_url": "https://heathercoxrichardson.substack.com",
        "source_key": "letters_from_an_american",
        "discovery_source": "handpicked_letters_from_an_american",
    },
    {
        "publication_url": "https://www.experimental-history.com",
        "source_key": "experimental_history",
        "discovery_source": "handpicked_experimental_history",
    },
    {
        "publication_url": "https://www.oneusefulthing.org",
        "source_key": "one_useful_thing",
        "discovery_source": "handpicked_one_useful_thing",
    },
    {
        "publication_url": "https://www.theintrinsicperspective.com",
        "source_key": "intrinsic_perspective",
        "discovery_source": "handpicked_intrinsic_perspective",
    },
    {
        "publication_url": "https://www.ageofinvention.xyz",
        "source_key": "age_of_invention",
        "discovery_source": "handpicked_age_of_invention",
    },
    
]

# Path segments that identify individual post/profile/note pages — not publication homepages.
_POST_PATH_SEGMENTS = ("/p/", "/profile/", "/notes/", "/comments")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_collection.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape public Substack publications for the CognLoad project."
    )
    p.add_argument("--input-csv", type=Path, default=None,
                   help="CSV with extra publication URLs (column: publication_url)")
    p.add_argument("--max-posts-per-publication", type=int, default=None,
                   help="Fallback cap for posts per publication if a source-specific cap is not set")
    p.add_argument("--refresh-url-index", action="store_true",
                   help="Rebuild posts_index.csv for publications in this run instead of skipping already-indexed publications")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"),
                   help="Directory for output CSVs (default: data/processed)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    p.add_argument("--fix-titles", action="store_true",
                   help="Re-extract og:title from saved raw HTML and patch posts_scraped.csv + article_features.csv, then exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_url(
    url: str,
    session: requests.Session,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[requests.Response | None, str | None]:
    """GET a URL with a polite sleep. Returns (response, error_string)."""
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response, None
    except requests.RequestException as e:
        return None, str(e)


def check_robots(base_url: str, session: requests.Session) -> bool:
    """Return True if our User-Agent is allowed to scrape base_url."""
    robots_url = base_url.rstrip("/") + "/robots.txt"
    response, error = fetch_url(robots_url, session)
    if error or response is None:
        log.debug("robots.txt unreachable for %s — assuming allowed", base_url)
        return True
    rp = RobotFileParser()
    rp.parse(response.text.splitlines())
    return rp.can_fetch(USER_AGENT, base_url.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def load_existing_progress(path: Path) -> pd.DataFrame:
    """Load CSV if it exists; return empty DataFrame otherwise."""
    if path.exists():
        log.debug("Resuming from %s", path)
        return pd.read_csv(path)
    return pd.DataFrame()


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.debug("Checkpoint saved: %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# URL collection
# ---------------------------------------------------------------------------

def _is_post_url(url: str, base_url: str) -> bool:
    """Return True if URL is a post on the same domain (has /p/ segment)."""
    parsed = urlparse(url)
    base_parsed = urlparse(base_url)
    return parsed.netloc == base_parsed.netloc and "/p/" in parsed.path


def extract_post_urls_from_sitemap(
    base_url: str, session: requests.Session
) -> list[tuple[str, str]]:
    """
    Fetch sitemap.xml and return list of (post_url, title).
    Handles sitemap index files by following the first child sitemap.
    Returns [] on any error.

    NOTE: If this returns [] for a real publication, it may mean the sitemap
    uses a different structure. Flag for discussion before patching.
    """
    sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
    response, error = fetch_url(sitemap_url, session)
    if error or response is None:
        log.debug("Sitemap fetch failed for %s: %s", base_url, error)
        return []

    soup = BeautifulSoup(response.content, "lxml-xml")

    # Handle sitemap index: follow first child sitemap
    if soup.find("sitemapindex"):
        child_loc = soup.find("sitemap")
        if child_loc:
            child_url = child_loc.find("loc")
            if child_url:
                child_response, child_error = fetch_url(child_url.get_text(strip=True), session)
                if child_error or child_response is None:
                    log.debug("Child sitemap fetch failed: %s", child_error)
                    return []
                soup = BeautifulSoup(child_response.content, "lxml-xml")

    results = []
    for url_tag in soup.find_all("url"):
        loc = url_tag.find("loc")
        if loc is None:
            continue
        url = loc.get_text(strip=True)
        if not _is_post_url(url, base_url):
            continue
        title_tag = url_tag.find("title") or url_tag.find("news:title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        results.append((url, title))

    log.info("Sitemap: found %d post URLs for %s", len(results), base_url)
    return results


def extract_post_urls_from_archive(
    base_url: str, session: requests.Session
) -> list[tuple[str, str]]:
    """
    Fallback: fetch /archive and extract /p/ links.
    Returns [] on any error.

    NOTE: Archive page structure varies. If this returns [] on real pages,
    flag for discussion — the selectors may need adjustment.
    """
    archive_url = base_url.rstrip("/") + "/archive"
    response, error = fetch_url(archive_url, session)
    if error or response is None:
        log.debug("Archive fetch failed for %s: %s", base_url, error)
        return []

    soup = BeautifulSoup(response.content, "lxml")
    results = []
    seen: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, a_tag["href"])
        if full_url in seen or not _is_post_url(full_url, base_url):
            continue
        seen.add(full_url)
        results.append((full_url, a_tag.get_text(strip=True)))

    log.info("Archive: found %d post URLs for %s", len(results), base_url)
    return results


# ---------------------------------------------------------------------------
# Article text extraction
# ---------------------------------------------------------------------------

# Substack post body selectors, tried in priority order.
# NOTE: If all return empty on real pages, the class names may differ.
# Flag for discussion before changing selectors.
_ARTICLE_SELECTORS = [
    "div.available-content",
    "div.post-content",
    "article",
    "div[class*='body']",
]

_NOISE_TAGS = ["nav", "footer", "script", "style", "button", "form",
               "aside", "figure", "figcaption"]
_NOISE_CLASSES = ["subscribe", "paywall", "comments", "share",
                  "header", "widget", "ad", "sidebar"]


def extract_article_text(soup: BeautifulSoup) -> str:
    """
    Extract main article body text from a BeautifulSoup page.
    Returns paragraph text joined by double newlines, or '' if nothing found.
    """
    container = None
    for selector in _ARTICLE_SELECTORS:
        container = soup.select_one(selector)
        if container:
            break

    if container is None:
        return ""

    for tag in _NOISE_TAGS:
        for el in container.find_all(tag):
            el.decompose()

    for el in container.find_all(class_=True):
        if not el.attrs:
            continue
        classes = " ".join(el.get("class", []))
        if any(noise in classes.lower() for noise in _NOISE_CLASSES):
            el.decompose()

    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for p in container.find_all("p")
        if p.get_text(strip=True)
    ]
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Engagement metrics extraction
# ---------------------------------------------------------------------------

def _parse_count(text: str | None) -> int | None:
    """Extract first integer from a string like '47 comments' -> 47."""
    if not text:
        return None
    m = re.search(r"\d[\d,]*", text)
    return int(m.group().replace(",", "")) if m else None


def _engagement_from_html(soup: BeautifulSoup) -> dict:
    """
    Layer 1: Read like/comment counts from visible HTML.

    NOTE: Selectors are based on expected Substack structure.
    If real pages consistently return None here, the class names differ —
    flag for discussion before patching selectors.
    """
    like_count = None
    comment_count = None

    for sel in ["span.like-count", "button.like-button span", "[class*='like-count']"]:
        el = soup.select_one(sel)
        if el:
            like_count = _parse_count(el.get_text(strip=True))
            if like_count is not None:
                break

    for sel in ["[class*='comment-button'] .label",
                "a[href*='comments'] span",
                "[class*='comment-count']"]:
        el = soup.select_one(sel)
        if el:
            comment_count = _parse_count(el.get_text(strip=True))
            if comment_count is not None:
                break

    return {"like_count": like_count, "comment_count": comment_count}


def _engagement_from_script_json(soup: BeautifulSoup) -> dict:
    """
    Layer 2: Parse engagement from embedded <script type="application/json">.
    Substack embeds post data in a block with id="__NEXT_DATA__".

    NOTE: JSON structure may vary across Substack versions — flag if parsing fails.
    """
    like_count = None
    comment_count = None

    for script in soup.find_all("script", type="application/json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        post = (
            data.get("props", {})
                .get("pageProps", {})
                .get("post", {})
        )
        if not post:
            continue
        reactions = post.get("reactions", {})
        if reactions:
            like_count = sum(reactions.values())
        comment_count = post.get("comment_count")
        if like_count is not None or comment_count is not None:
            break

    return {"like_count": like_count, "comment_count": comment_count}


def _engagement_from_ld_json(soup: BeautifulSoup) -> dict:
    """
    Layer 2: Parse engagement from <script type="application/ld+json"> (Schema.org).
    Substack embeds interactionStatistic with LikeAction and CommentAction counts.
    Works on both older (__NEXT_DATA__) and newer (ES module) Substack frontends.
    """
    like_count = None
    comment_count = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stats = data.get("interactionStatistic", [])
        for stat in stats:
            itype = stat.get("interactionType", "")
            count = stat.get("userInteractionCount")
            if count is None:
                continue
            if "LikeAction" in itype:
                like_count = int(count)
            elif "CommentAction" in itype:
                comment_count = int(count)
        if like_count is not None or comment_count is not None:
            break
    return {"like_count": like_count, "comment_count": comment_count}


def extract_engagement_metrics(
    soup: BeautifulSoup,
    post_url: str,
    session: requests.Session,
) -> dict:
    """
    Layered engagement extraction.
    Always returns: like_count, comment_count, engagement_available, engagement_source.
    """
    # Layer 1: visible HTML
    result = _engagement_from_html(soup)
    if result["like_count"] is not None or result["comment_count"] is not None:
        return {**result, "engagement_available": True, "engagement_source": "visible_html"}

    # Layer 2: application/ld+json (Schema.org interactionStatistic)
    result = _engagement_from_ld_json(soup)
    if result["like_count"] is not None or result["comment_count"] is not None:
        return {**result, "engagement_available": True, "engagement_source": "ld_json"}

    # Layer 3: embedded script JSON (__NEXT_DATA__)
    result = _engagement_from_script_json(soup)
    if result["like_count"] is not None or result["comment_count"] is not None:
        return {**result, "engagement_available": True, "engagement_source": "script_json"}

    # Layer 4: public comments page
    comments_url = post_url.rstrip("/") + "/comments"
    response, error = fetch_url(comments_url, session)
    if error is None and response is not None:
        csoup = BeautifulSoup(response.content, "lxml")
        html_result = _engagement_from_html(csoup)
        if html_result["comment_count"] is not None:
            return {
                "like_count": None,
                "comment_count": html_result["comment_count"],
                "engagement_available": True,
                "engagement_source": "comments_page",
            }

    return {
        "like_count": None,
        "comment_count": None,
        "engagement_available": False,
        "engagement_source": "missing",
    }


# ---------------------------------------------------------------------------
# Publication verification
# ---------------------------------------------------------------------------

def verify_publication_engagement(
    post_urls: list[str],
    session: requests.Session,
    n: int = ENGAGEMENT_TEST_N,
) -> tuple[bool, int, int]:
    """
    Test first N post URLs for visible engagement metrics.
    Returns (passed, tested_count, found_count).
    passed = True if at least 1 post has any engagement metric.
    """
    sample = post_urls[:n]
    found_count = 0

    for url in sample:
        response, error = fetch_url(url, session)
        if error or response is None:
            log.debug("Engagement check: failed to fetch %s: %s", url, error)
            continue
        soup = BeautifulSoup(response.content, "lxml")
        metrics = extract_engagement_metrics(soup, url, session)
        if metrics["engagement_available"]:
            found_count += 1

    tested = len(sample)
    passed = found_count > 0
    log.info(
        "Engagement check: %d/%d posts had metrics (passed=%s)",
        found_count, tested, passed,
    )
    return passed, tested, found_count


def post_cap_for_publication(pub: pd.Series | dict, args: argparse.Namespace) -> int | None:
    """Return the post cap for a publication row."""
    return args.max_posts_per_publication

# ---------------------------------------------------------------------------
# Phase 1: Publication discovery
# ---------------------------------------------------------------------------

def run_phase1_discovery(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """
    Build publications_verified.csv from:
      1. handpicked seeds,
      2. optional input CSV rows,
      3. optional popular Substack discovery.
    Skips publications already present in the checkpoint file.
    """
    pub_path = output_dir / "publications_verified.csv"
    existing = load_existing_progress(pub_path)
    done_urls = set(existing["publication_url"].tolist()) if "publication_url" in existing.columns else set()

    session = _make_session()

    publication_candidates: list[dict] = list(SEED_PUBLICATIONS)

    if args.input_csv and args.input_csv.exists():
        extra = pd.read_csv(args.input_csv)
        if "publication_url" not in extra.columns:
            log.warning("--input-csv missing 'publication_url' column — skipping")
        else:
            for pub_url in extra["publication_url"].dropna().tolist():
                publication_candidates.append({
                    "publication_url": str(pub_url),
                    "source_key": "csv_input",
                    "discovery_source": "csv_input",
                })

    # Deduplicate preserving order across seed, CSV, and popular sources.
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in publication_candidates:
        u = str(item["publication_url"]).strip().rstrip("/")
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append({**item, "publication_url": u})

    rows = existing.to_dict("records") if len(existing) else []

    for item in tqdm(deduped, desc="Phase 1: publications"):
        pub_url = item["publication_url"]
        if pub_url in done_urls:
            log.info("Skipping already-processed publication: %s", pub_url)
            continue

        log.info("Checking publication: %s", pub_url)
        row = {
            "publication_url": pub_url,
            "publication_name": urlparse(pub_url).netloc,
            "source_key": item.get("source_key", "unknown"),
            "discovery_source": item.get("discovery_source", "unknown"),
            "popular_discovery_url": item.get("popular_discovery_url", ""),
            "robots_allowed": None,
            "engagement_check_passed": None,
            "scrape_status": "pending",
            "post_count": 0,
            "post_cap": post_cap_for_publication(item, args),
            "notes": "",
        }

        allowed = check_robots(pub_url, session)
        row["robots_allowed"] = allowed
        if not allowed:
            log.warning("robots.txt disallows scraping: %s", pub_url)
            row["scrape_status"] = "failed_robots"
            rows.append(row)
            save_checkpoint(pd.DataFrame(rows), pub_path)
            continue

        row["scrape_status"] = "success"
        rows.append(row)
        save_checkpoint(pd.DataFrame(rows), pub_path)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Phase 2: URL collection
# ---------------------------------------------------------------------------

def run_phase2_url_collection(
    publications_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Collect post URLs for each approved publication via sitemap or archive.
    Runs engagement verification before queuing a publication for scraping.
    """
    posts_path = output_dir / "posts_index.csv"
    pub_path = output_dir / "publications_verified.csv"

    existing_posts = load_existing_progress(posts_path)

    if args.refresh_url_index and len(existing_posts) and "publication_url" in existing_posts.columns:
        current_pub_urls = set(publications_df["publication_url"].dropna().tolist())
        existing_posts = existing_posts[~existing_posts["publication_url"].isin(current_pub_urls)]
        log.info(
            "Refreshing URL index for %d current publications; keeping %d existing rows from other publications",
            len(current_pub_urls),
            len(existing_posts),
        )

    done_pub_urls = (
        set(existing_posts["publication_url"].unique().tolist())
        if "publication_url" in existing_posts.columns
        else set()
    )

    all_post_rows = existing_posts.to_dict("records") if len(existing_posts) else []
    pub_rows = publications_df.to_dict("records")
    pub_rows_by_url = {r["publication_url"]: r for r in pub_rows}
    session = _make_session()

    eligible = publications_df[publications_df["scrape_status"] == "success"]

    for _, pub in tqdm(eligible.iterrows(), total=len(eligible), desc="Phase 2: URL collection"):
        pub_url = pub["publication_url"]
        if pub_url in done_pub_urls:
            log.info("Skipping URL collection (already done): %s", pub_url)
            continue

        log.info("Collecting URLs: %s", pub_url)

        post_urls = extract_post_urls_from_sitemap(pub_url, session)
        url_source = "sitemap"
        if not post_urls:
            log.info("Sitemap empty or failed — trying archive for %s", pub_url)
            post_urls = extract_post_urls_from_archive(pub_url, session)
            url_source = "archive"

        pub_row = pub_rows_by_url.get(pub_url)

        if not post_urls:
            log.warning("No post URLs found for %s", pub_url)
            if pub_row:
                pub_row["scrape_status"] = "failed_parse"
                pub_row["notes"] = "No post URLs found via sitemap or archive"
            save_checkpoint(pd.DataFrame(pub_rows), pub_path)
            continue

        post_cap = post_cap_for_publication(pub, args)

        if post_cap == 0:
            log.info("Skipping %s because its post cap is 0", pub_url)
            if pub_row:
                pub_row["scrape_status"] = "skipped_cap_zero"
                pub_row["engagement_check_passed"] = None
                pub_row["post_count"] = 0
                pub_row["post_cap"] = 0
                pub_row["notes"] = "Skipped because post cap was set to 0"
            save_checkpoint(pd.DataFrame(pub_rows), pub_path)
            continue

        url_list = [u for u, _ in post_urls]
        passed, tested, found = verify_publication_engagement(url_list, session)

        if not passed:
            log.warning(
                "Engagement check failed for %s (%d/%d had metrics) — skipping",
                pub_url, found, tested,
            )
            if pub_row:
                pub_row["scrape_status"] = "failed_engagement_check"
                pub_row["engagement_check_passed"] = False
                pub_row["notes"] = f"0/{tested} test posts had visible engagement metrics"
            save_checkpoint(pd.DataFrame(pub_rows), pub_path)
            continue

        if pub_row:
            pub_row["engagement_check_passed"] = True
            pub_row["post_count"] = len(post_urls)
            pub_row["post_cap"] = post_cap

        for url, title in post_urls:
            all_post_rows.append({
                "post_url": url,
                "title": title,
                "publication_url": pub_url,
                "publication_name": pub["publication_name"],
                "url_source": url_source,
                "source_key": pub.get("source_key", "unknown"),
                "discovery_source": pub.get("discovery_source", "unknown"),
                "popular_discovery_url": pub.get("popular_discovery_url", ""),
                "post_cap": post_cap,
                "scrape_status": "pending",
            })

        save_checkpoint(pd.DataFrame(all_post_rows), posts_path)
        save_checkpoint(pd.DataFrame(pub_rows), pub_path)
        log.info("Queued %d posts for %s", len(post_urls), pub_url)

    return pd.DataFrame(all_post_rows)


# ---------------------------------------------------------------------------
# Phase 3: Scraping
# ---------------------------------------------------------------------------

def extract_og_title(html_or_soup) -> str | None:
    """Return og:title content from raw HTML string or an existing BeautifulSoup."""
    if isinstance(html_or_soup, str):
        html_or_soup = BeautifulSoup(html_or_soup, "html.parser")
    tag = html_or_soup.find("meta", property="og:title")
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None

RAW_HTML_DIR = Path("data/raw_html")


def _url_to_slug(url: str) -> str:
    return re.sub(r"[^\w\-]", "_", urlparse(url).path.strip("/"))


def scrape_post(
    post_url: str,
    publication_name: str,
    session: requests.Session,
) -> dict:
    """
    Fetch and extract one post. Returns a dict with all posts_scraped.csv fields.
    On failure, returns a row with the appropriate scrape_status.
    """
    row: dict = {
        "post_url": post_url,
        "publication_name": publication_name,
        "title": None,
        "publish_date": None,
        "word_count": None,
        "paragraph_count": None,
        "like_count": None,
        "comment_count": None,
        "engagement_available": False,
        "engagement_source": "missing",
        "raw_html_path": None,
        "scrape_status": "pending",
        "notes": "",
    }

    response, error = fetch_url(post_url, session)
    if error or response is None:
        row["scrape_status"] = "failed_request"
        row["notes"] = error or "No response"
        log.warning("Request failed: %s — %s", post_url, error)
        return row

    soup = BeautifulSoup(response.content, "lxml")

    try:
        pub_slug = re.sub(r"[^\w\-]", "_", publication_name)
        html_dir = RAW_HTML_DIR / pub_slug
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_dir / f"{_url_to_slug(post_url)}.html"
        html_path.write_bytes(response.content)
        row["raw_html_path"] = str(html_path)
    except Exception as e:
        log.debug("Could not save raw HTML for %s: %s", post_url, e)

    og_title = extract_og_title(soup)
    if og_title:
        row["title"] = og_title
    else:
        h1 = soup.find("h1")
        h1_text = h1.get_text(strip=True) if h1 else ""
        if h1_text:
            row["title"] = h1_text
        else:
            title_tag = soup.find("title")
            row["title"] = title_tag.get_text(strip=True) if title_tag else None

    publish_date = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            raw = data.get("datePublished")
            if raw:
                publish_date = str(dateparser.parse(raw).date())
                break
        except Exception:
            continue
    if not publish_date:
        date_tag = (
            soup.find("time")
            or soup.select_one("[class*='date']")
            or soup.select_one("[class*='post-date']")
        )
        if date_tag:
            date_text = date_tag.get("datetime") or date_tag.get_text(strip=True)
            try:
                publish_date = str(dateparser.parse(date_text).date())
            except Exception:
                pass
    row["publish_date"] = publish_date

    article_text = extract_article_text(soup)
    if not article_text:
        row["scrape_status"] = "failed_parse"
        row["notes"] = "extract_article_text returned empty — HTML structure may differ from expected"
        log.warning("Empty article text for %s — flag for discussion", post_url)
        return row

    paragraphs = segment_paragraphs(article_text)
    row["word_count"] = len(article_text.split())
    row["paragraph_count"] = len(paragraphs)

    # Log quality warning (do not discard — keep for text analysis even if short)
    if not validate_post_row(row):
        log.debug(
            "Post below quality threshold (word_count=%s, para_count=%s): %s",
            row["word_count"], row["paragraph_count"], post_url,
        )

    row.update(extract_engagement_metrics(soup, post_url, session))

    row["scrape_status"] = "success"
    return row


def run_phase3_scraping(
    posts_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    """Scrape each pending post. Checkpoints after every post."""
    scraped_path = output_dir / "posts_scraped.csv"
    existing = load_existing_progress(scraped_path)
    done_urls = (
        set(existing["post_url"].tolist())
        if "post_url" in existing.columns
        else set()
    )

    scraped_rows = existing.to_dict("records") if len(existing) else []
    session = _make_session()

    # Seed valid-post counts from already-scraped data so resume works correctly.
    valid_counts: dict[str, int] = {}
    if len(existing) and "publication_url" in existing.columns and "scrape_status" in existing.columns:
        for _, er in existing[existing["scrape_status"] == "success"].iterrows():
            if validate_post_row(er.to_dict()):
                pub = er.get("publication_url", "")
                valid_counts[pub] = valid_counts.get(pub, 0) + 1

    pending = posts_df[
        (posts_df["scrape_status"] == "pending")
        & (~posts_df["post_url"].isin(done_urls))
    ] if len(posts_df) else pd.DataFrame()

    log.info("Posts to scrape: %d", len(pending))

    for _, post in tqdm(pending.iterrows(), total=len(pending), desc="Phase 3: scraping"):
        url = post["post_url"]
        pub_url = post.get("publication_url", "")
        post_cap = post.get("post_cap")

        if post_cap is not None and valid_counts.get(pub_url, 0) >= post_cap:
            log.debug(
                "Skipping %s — %s already has %d valid posts (cap=%d)",
                url, pub_url, valid_counts[pub_url], post_cap,
            )
            continue

        row = scrape_post(url, post.get("publication_name", "unknown"), session)
        # Carry over index fields so scraped outputs keep source/provenance metadata.
        for field in [
            "publication_url",
            "publication_name",
            "url_source",
            "source_key",
            "discovery_source",
            "popular_discovery_url",
            "post_cap",
        ]:
            if row.get(field) is None and field in post:
                row[field] = post[field]

        if row["scrape_status"] == "success" and validate_post_row(row):
            valid_counts[pub_url] = valid_counts.get(pub_url, 0) + 1

        scraped_rows.append(row)
        done_urls.add(url)
        save_checkpoint(pd.DataFrame(scraped_rows), scraped_path)

    df = pd.DataFrame(scraped_rows) if scraped_rows else pd.DataFrame()
    for col in [
        "like_count",
        "comment_count",
        "engagement_available",
        "engagement_source",
        "source_key",
        "discovery_source",
        "popular_discovery_url",
        "post_cap",
    ]:
        if col not in df.columns:
            df[col] = None

    if len(df):
        validate_dataframe_columns(
            df,
            ["post_url", "like_count", "comment_count", "engagement_available", "engagement_source"],
        )
    return df


# ---------------------------------------------------------------------------
# Text segmentation and validation
# ---------------------------------------------------------------------------

def segment_paragraphs(text: str) -> list[str]:
    """Split on blank lines; discard paragraphs below MIN_PARAGRAPH_WORDS."""
    if not text:
        return []
    raw = re.split(r"\n{2,}", text)
    return [
        p.strip()
        for p in raw
        if p.strip() and len(p.split()) >= MIN_PARAGRAPH_WORDS
    ]


def validate_post_row(row: dict) -> bool:
    """Return True if post meets minimum length requirements."""
    return (
        row.get("word_count", 0) >= MIN_WORD_COUNT
        and row.get("paragraph_count", 0) >= MIN_PARAGRAPH_COUNT
    )


def validate_dataframe_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError listing any required columns absent from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")


# ---------------------------------------------------------------------------
# Phase 4: Paragraph segmentation
# ---------------------------------------------------------------------------

def run_phase4_segmentation(scraped_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Build paragraphs.csv from successfully scraped posts, re-reading saved HTML."""
    para_path = output_dir / "paragraphs.csv"
    existing = load_existing_progress(para_path)
    done_urls = (
        set(existing["post_url"].tolist())
        if "post_url" in existing.columns
        else set()
    )

    para_rows = existing.to_dict("records") if len(existing) else []
    success = scraped_df[scraped_df["scrape_status"] == "success"] if len(scraped_df) else pd.DataFrame()

    for _, post in tqdm(success.iterrows(), total=len(success), desc="Phase 4: segmenting"):
        url = post["post_url"]
        if url in done_urls:
            continue

        article_text = ""
        html_path = post.get("raw_html_path")
        if html_path:
            try:
                soup = BeautifulSoup(Path(html_path).read_bytes(), "lxml")
                article_text = extract_article_text(soup)
            except OSError:
                pass

        for i, para in enumerate(segment_paragraphs(article_text)):
            para_rows.append({
                "post_url": url,
                "paragraph_index": i,
                "paragraph_text": para,
                "word_count": len(para.split()),
            })
        done_urls.add(url)

    df = pd.DataFrame(para_rows) if para_rows else pd.DataFrame(
        columns=["post_url", "paragraph_index", "paragraph_text", "word_count"]
    )
    save_checkpoint(df, para_path)
    return df


# ---------------------------------------------------------------------------
# Phase 5: Summary
# ---------------------------------------------------------------------------

def run_phase5_summary(
    publications_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    scraped_df: pd.DataFrame,
    paragraphs_df: pd.DataFrame,
    start_time: datetime,
) -> None:
    """Write scrape_summary.csv and print terminal summary."""
    end_time = datetime.now()
    runtime_secs = (end_time - start_time).total_seconds()

    n_pubs = len(publications_df)
    n_posts_found = len(posts_df)
    n_ok = int((scraped_df["scrape_status"] == "success").sum()) if len(scraped_df) else 0
    n_failed = int(scraped_df["scrape_status"].str.startswith("failed").sum()) if len(scraped_df) else 0
    n_with_engagement = int(scraped_df["engagement_available"].sum()) if "engagement_available" in scraped_df.columns and len(scraped_df) else 0
    engagement_rate = (n_with_engagement / n_ok * 100) if n_ok else 0.0

    summary = {
        "run_start": start_time.isoformat(),
        "run_end": end_time.isoformat(),
        "runtime_seconds": round(runtime_secs, 1),
        "publications_processed": n_pubs,
        "posts_found": n_posts_found,
        "posts_scraped_success": n_ok,
        "posts_failed": n_failed,
        "posts_with_engagement": n_with_engagement,
        "engagement_availability_pct": round(engagement_rate, 1),
        "paragraphs_total": len(paragraphs_df),
    }

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(outputs_dir / "scrape_summary.csv", index=False)

    log.info("=== Run complete ===")
    log.info(
        "Publications: %d | Posts found: %d | Scraped OK: %d | Failed: %d",
        n_pubs, n_posts_found, n_ok, n_failed,
    )
    log.info("Engagement: %d posts (%.1f%%)", n_with_engagement, engagement_rate)

    print("\n" + "=" * 60)
    print("Data Collection Summary")
    print("=" * 60)
    print(f"  Runtime:            {runtime_secs:.0f}s")
    print(f"  Publications:       {n_pubs}")
    print(f"  Posts found:        {n_posts_found}")
    print(f"  Scraped (success):  {n_ok}")
    print(f"  Failed:             {n_failed}")
    print(f"  With engagement:    {n_with_engagement} ({engagement_rate:.1f}%)")
    print(f"  Paragraphs:         {len(paragraphs_df)}")
    print("=" * 60)

    if len(scraped_df) and "publication_url" in scraped_df.columns:
        print("\nPer-Publication Breakdown")
        print("-" * 60)
        print(f"  {'Publication':<35} {'Posts':>6} {'Paragraphs':>11}")
        print(f"  {'-'*35} {'------':>6} {'-----------':>11}")

        # Map post_url -> publication for paragraph counting
        para_pub_map: dict[str, str] = {}
        if len(scraped_df):
            para_pub_map = dict(zip(scraped_df["post_url"], scraped_df["publication_url"]))

        para_counts: dict[str, int] = {}
        if len(paragraphs_df) and "post_url" in paragraphs_df.columns:
            for post_url, pub_url in para_pub_map.items():
                para_counts[pub_url] = para_counts.get(pub_url, 0) + int(
                    (paragraphs_df["post_url"] == post_url).sum()
                )

        success_df = scraped_df[scraped_df["scrape_status"] == "success"]
        pub_post_counts = success_df.groupby("publication_url").size().to_dict()

        all_pub_urls = list(
            dict.fromkeys(
                [p["publication_url"] for p in SEED_PUBLICATIONS]
                + list(pub_post_counts.keys())
            )
        )

        for pub_url in all_pub_urls:
            label = urlparse(pub_url).netloc
            posts_count = pub_post_counts.get(pub_url, 0)
            paras_count = para_counts.get(pub_url, 0)
            print(f"  {label:<35} {posts_count:>6} {paras_count:>11}")

        print("-" * 60)
    print()


def run_fix_titles(output_dir: Path) -> None:
    """Re-extract og:title from saved raw HTML and patch posts_scraped.csv + article_features.csv."""
    scraped_path = output_dir / "posts_scraped.csv"
    features_path = output_dir / "article_features.csv"

    scraped = pd.read_csv(scraped_path)
    fixed = 0
    for i, row in scraped.iterrows():
        path = row.get("raw_html_path")
        if not path:
            continue
        html_path = Path(path)
        if not html_path.exists():
            continue
        og_title = extract_og_title(html_path.read_text(encoding="utf-8", errors="ignore"))
        if og_title and og_title != row["title"]:
            scraped.at[i, "title"] = og_title
            fixed += 1

    print(f"Fixed {fixed} titles in posts_scraped.csv")
    scraped.to_csv(scraped_path, index=False)

    features = pd.read_csv(features_path)
    title_map = scraped.set_index("post_url")["title"].to_dict()
    before = features["title"].copy()
    features["title"] = features["post_url"].map(title_map).fillna(features["title"])
    changed = (features["title"] != before).sum()
    print(f"Fixed {changed} titles in article_features.csv")
    features.to_csv(features_path, index=False)


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)
    start_time = datetime.now()
    log.info("Data collection started at %s", start_time.isoformat())
    log.info("Args: %s", vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fix_titles:
        run_fix_titles(args.output_dir)
        return

    publications_df = run_phase1_discovery(args, args.output_dir)
    posts_df = run_phase2_url_collection(publications_df, args, args.output_dir)
    scraped_df = run_phase3_scraping(posts_df, args, args.output_dir)
    paragraphs_df = run_phase4_segmentation(scraped_df, args.output_dir)
    run_phase5_summary(publications_df, posts_df, scraped_df, paragraphs_df, start_time)

    log.info("Data collection finished at %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
