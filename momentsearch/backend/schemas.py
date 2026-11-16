from typing import List, Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    status: str  # uploaded | processing | completed | failed
    progress: float = 0.0
    duration_sec: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    error_message: Optional[str] = None
    created_at: str


class ProcessRequest(BaseModel):
    sampling_fps: float = Field(default=1.0, ge=0.25, le=5.0)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    top_k: int = Field(default=5, ge=1, le=25)
    merge_gap_sec: float = Field(default=2.0, ge=0.0, le=10.0)


class SearchMatch(BaseModel):
    rank: int
    start_time: float
    end_time: float
    peak_timestamp: float
    similarity_score: float
    thumbnail_url: str


class SearchResponse(BaseModel):
    video_id: str
    query: str
    disclaimer: str
    results: List[SearchMatch]