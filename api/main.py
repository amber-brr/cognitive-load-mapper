import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from api import data, rewrite
from api.schemas import (
    ArticleDetail, ArticleSummary, FlaggedParagraph,
    ParagraphInfo, RewriteRequest, RewriteResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        data.load()
    except Exception as e:
        logger.error("Data load failed: %s", e)
    yield


app = FastAPI(
    title="Cognitive Complexity Gradient API",
    description="Cognitive load analysis for long-form articles.",
    version="1.0.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
@limiter.limit("100/minute")
def health(request: Request):
    return {"status": "ok"}


@app.get("/ready")
@limiter.limit("100/minute")
def ready(request: Request):
    if data.articles.empty:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    return {"status": "ready", "articles": len(data.articles)}


@app.get("/articles", response_model=list[ArticleSummary])
@limiter.limit("30/minute")
def list_articles(
    request: Request,
    publication: Optional[str] = None,
    shape: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    return data.list_articles(publication, shape, limit, offset)


@app.get("/articles/{article_id}", response_model=ArticleDetail)
@limiter.limit("30/minute")
def get_article(request: Request, article_id: int):
    row, paras = data.get_article(article_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found.")
    return ArticleDetail(
        article_id=article_id,
        title=row.get("title") or "",
        publication_name=row.get("publication_name") or "",
        post_url=row.get("post_url") or "",
        publish_date=str(row["publish_date"]) if row.get("publish_date") else None,
        word_count=int(row["word_count"]) if row.get("word_count") is not None else None,
        gradient_shape=row.get("gradient_shape") or "",
        mean_complexity=float(row.get("mean_complexity") or 0),
        complexity_variance=float(row.get("complexity_variance") or 0),
        complexity_slope=float(row.get("complexity_slope") or 0),
        peak_position=float(row.get("peak_position") or 0),
        resolution_index=float(row.get("resolution_index") or 0),
        engagement_z=float(row["engagement_z"]) if row.get("engagement_z") is not None else None,
        paragraphs=[ParagraphInfo(**p) for p in paras],
    )


@app.post("/articles/{article_id}/rewrite", response_model=RewriteResponse)
@limiter.limit("10/minute")
def rewrite_article(request: Request, article_id: int, body: RewriteRequest):
    row, paras = data.get_article(article_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found.")

    current_shape = row.get("gradient_shape") or ""
    target_shape = body.target_shape

    if current_shape == target_shape:
        return RewriteResponse(
            article_id=article_id,
            current_shape=current_shape,
            target_shape=target_shape,
            flagged_paragraphs=[],
            message="Article already has the target gradient shape.",
        )

    flagged = data.flag_paragraphs(paras, target_shape)
    mean_complexity = float(row.get("mean_complexity") or 0)

    result = []
    for p in flagged:
        direction = rewrite.target_direction(p["complexity_v1"], mean_complexity)
        rewritten = rewrite.rewrite_paragraph(p["paragraph_text"], direction, p["reason"])
        result.append(
            FlaggedParagraph(
                paragraph_index=p["paragraph_index"],
                paragraph_text=p["paragraph_text"],
                complexity_v1=p["complexity_v1"],
                reason=p["reason"],
                rewritten_text=rewritten,
            )
        )

    logger.info("Rewrite: article=%d %s→%s, %d paragraphs flagged.", article_id, current_shape, target_shape, len(result))
    return RewriteResponse(
        article_id=article_id,
        current_shape=current_shape,
        target_shape=target_shape,
        flagged_paragraphs=result,
    )
