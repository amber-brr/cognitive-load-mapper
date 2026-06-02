from typing import Literal, Optional
from pydantic import BaseModel


class ArticleSummary(BaseModel):
    article_id: int
    title: str
    publication_name: str
    gradient_shape: str
    word_count: Optional[int]
    mean_complexity: float
    engagement_z: Optional[float]


class ParagraphInfo(BaseModel):
    paragraph_index: int
    paragraph_text: str
    complexity_v1: float
    paragraph_position_norm: float


class ArticleDetail(BaseModel):
    article_id: int
    title: str
    publication_name: str
    post_url: str
    publish_date: Optional[str]
    word_count: Optional[int]
    gradient_shape: str
    mean_complexity: float
    complexity_variance: float
    complexity_slope: float
    peak_position: float
    resolution_index: float
    engagement_z: Optional[float]
    paragraphs: list[ParagraphInfo]


class RewriteRequest(BaseModel):
    target_shape: Literal["ramp", "cliff", "plateau", "rollercoaster", "resolution"]


class FlaggedParagraph(BaseModel):
    paragraph_index: int
    paragraph_text: str
    complexity_v1: float
    reason: str
    rewritten_text: str


class RewriteResponse(BaseModel):
    article_id: int
    current_shape: str
    target_shape: str
    flagged_paragraphs: list[FlaggedParagraph]
    message: Optional[str] = None
