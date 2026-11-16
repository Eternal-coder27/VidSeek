from typing import Any, Dict, List
import faiss
import numpy as np
from config import INDEXES_DIR
import database as db


def save_video_index(video_id: str, index: faiss.IndexFlatIP) -> None:
    index_path = INDEXES_DIR / f"{video_id}.faiss"
    faiss.write_index(index, str(index_path))


def search_video_index(
    video_id: str,
    query_vec: np.ndarray,
    top_k: int = 5,
    merge_gap_sec: float = 2.0,
    sampling_fps: float = 1.0,
) -> List[Dict[str, Any]]:
    index_path = INDEXES_DIR / f"{video_id}.faiss"
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found. Process the video first.")

    index = faiss.read_index(str(index_path))
    frame_records = db.get_frame_records(video_id)
    if not frame_records or index.ntotal == 0:
        return []

    candidate_k = min(index.ntotal, max(top_k * 4, 15))
    scores, indices = index.search(query_vec, candidate_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(frame_records):
            rec = frame_records[idx]
            candidates.append(
                {
                    "timestamp": float(rec["timestamp_sec"]),
                    "score": float(score),
                    "thumbnail_name": rec["thumbnail_name"],
                }
            )

    # Sort chronologically and merge adjacent matching frames into [start_time, end_time] events
    candidates.sort(key=lambda x: x["timestamp"])
    frame_step = 1.0 / max(sampling_fps, 0.1)
    merged: List[Dict[str, Any]] = []

    for item in candidates:
        if not merged:
            merged.append(
                {
                    "start_time": item["timestamp"],
                    "end_time": item["timestamp"] + frame_step,
                    "peak_timestamp": item["timestamp"],
                    "similarity_score": item["score"],
                    "thumbnail_name": item["thumbnail_name"],
                }
            )
        else:
            last = merged[-1]
            if item["timestamp"] - last["end_time"] <= merge_gap_sec:
                last["end_time"] = max(last["end_time"], item["timestamp"] + frame_step)
                if item["score"] > last["similarity_score"]:
                    last["similarity_score"] = item["score"]
                    last["peak_timestamp"] = item["timestamp"]
                    last["thumbnail_name"] = item["thumbnail_name"]
            else:
                merged.append(
                    {
                        "start_time": item["timestamp"],
                        "end_time": item["timestamp"] + frame_step,
                        "peak_timestamp": item["timestamp"],
                        "similarity_score": item["score"],
                        "thumbnail_name": item["thumbnail_name"],
                    }
                )

    merged.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_intervals = merged[:top_k]

    results = []
    for rank, interval in enumerate(top_intervals, start=1):
        results.append(
            {
                "rank": rank,
                "start_time": round(interval["start_time"], 2),
                "end_time": round(interval["end_time"], 2),
                "peak_timestamp": round(interval["peak_timestamp"], 2),
                "similarity_score": round(interval["similarity_score"], 4),
                "thumbnail_url": f"/api/videos/{video_id}/frames/{interval['thumbnail_name']}",
            }
        )
    return results