import sqlite3
from typing import Any, Dict, List, Optional
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_ext TEXT NOT NULL,
                pipeline TEXT NOT NULL DEFAULT 'v1',
                status TEXT NOT NULL DEFAULT 'uploaded',
                progress REAL NOT NULL DEFAULT 0.0,
                duration_sec REAL NOT NULL DEFAULT 0.0,
                fps REAL NOT NULL DEFAULT 0.0,
                frame_count INTEGER NOT NULL DEFAULT 0,
                sampling_fps REAL NOT NULL DEFAULT 1.0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Safe migration if videos table already existed without 'pipeline'
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(videos)").fetchall()]
        if "pipeline" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'v1'")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                thumbnail_name TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_metadata_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                thumbnail_name TEXT NOT NULL,
                sharpness REAL NOT NULL DEFAULT 0.0,
                objects_json TEXT NOT NULL DEFAULT '[]',
                ocr_text TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
            """
        )
        conn.commit()


def create_video(video_id: str, filename: str, stored_ext: str, pipeline: str = "v1") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO videos (video_id, filename, stored_ext, pipeline, status) VALUES (?, ?, ?, ?, 'uploaded')",
            (video_id, filename, stored_ext, pipeline),
        )
        conn.commit()


def update_video_status(
    video_id: str,
    status: str,
    progress: float = 0.0,
    duration_sec: Optional[float] = None,
    fps: Optional[float] = None,
    frame_count: Optional[int] = None,
    sampling_fps: Optional[float] = None,
    pipeline: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE videos
            SET status = ?,
                progress = ?,
                duration_sec = COALESCE(?, duration_sec),
                fps = COALESCE(?, fps),
                frame_count = COALESCE(?, frame_count),
                sampling_fps = COALESCE(?, sampling_fps),
                pipeline = COALESCE(?, pipeline),
                error_message = ?
            WHERE video_id = ?
            """,
            (status, progress, duration_sec, fps, frame_count, sampling_fps, pipeline, error_message, video_id),
        )
        conn.commit()


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None


def list_videos() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def save_frame_records(video_id: str, records: List[tuple]) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM frame_metadata WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO frame_metadata (video_id, faiss_idx, timestamp_sec, thumbnail_name)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def get_frame_records(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faiss_idx, timestamp_sec, thumbnail_name
            FROM frame_metadata
            WHERE video_id = ?
            ORDER BY faiss_idx ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_frame_records_v2(video_id: str, records: List[tuple]) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM frame_metadata_v2 WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO frame_metadata_v2
                (video_id, faiss_idx, timestamp_sec, thumbnail_name, sharpness, objects_json, ocr_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def get_frame_records_v2(video_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faiss_idx, timestamp_sec, thumbnail_name, sharpness, objects_json, ocr_text
            FROM frame_metadata_v2
            WHERE video_id = ?
            ORDER BY faiss_idx ASC
            """,
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]

# import sqlite3
# from typing import Any, Dict, List, Optional
# from config import DB_PATH


# def get_conn() -> sqlite3.Connection:
#     DB_PATH.parent.mkdir(parents=True, exist_ok=True)
#     conn = sqlite3.connect(str(DB_PATH))
#     conn.row_factory = sqlite3.Row
#     return conn


# def init_db() -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS videos (
#                 video_id TEXT PRIMARY KEY,
#                 filename TEXT NOT NULL,
#                 stored_ext TEXT NOT NULL,
#                 status TEXT NOT NULL DEFAULT 'uploaded',
#                 progress REAL NOT NULL DEFAULT 0.0,
#                 duration_sec REAL NOT NULL DEFAULT 0.0,
#                 fps REAL NOT NULL DEFAULT 0.0,
#                 frame_count INTEGER NOT NULL DEFAULT 0,
#                 sampling_fps REAL NOT NULL DEFAULT 1.0,
#                 error_message TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """
#         )
#         conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS frame_metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 video_id TEXT NOT NULL,
#                 faiss_idx INTEGER NOT NULL,
#                 timestamp_sec REAL NOT NULL,
#                 thumbnail_name TEXT NOT NULL,
#                 FOREIGN KEY (video_id) REFERENCES videos(video_id)
#             )
#             """
#         )
#         conn.commit()


# def create_video(video_id: str, filename: str, stored_ext: str) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             "INSERT INTO videos (video_id, filename, stored_ext, status) VALUES (?, ?, ?, 'uploaded')",
#             (video_id, filename, stored_ext),
#         )
#         conn.commit()


# def update_video_status(
#     video_id: str,
#     status: str,
#     progress: float = 0.0,
#     duration_sec: Optional[float] = None,
#     fps: Optional[float] = None,
#     frame_count: Optional[int] = None,
#     sampling_fps: Optional[float] = None,
#     error_message: Optional[str] = None,
# ) -> None:
#     with get_conn() as conn:
#         conn.execute(
#             """
#             UPDATE videos
#             SET status = ?,
#                 progress = ?,
#                 duration_sec = COALESCE(?, duration_sec),
#                 fps = COALESCE(?, fps),
#                 frame_count = COALESCE(?, frame_count),
#                 sampling_fps = COALESCE(?, sampling_fps),
#                 error_message = ?
#             WHERE video_id = ?
#             """,
#             (status, progress, duration_sec, fps, frame_count, sampling_fps, error_message, video_id),
#         )
#         conn.commit()


# def get_video(video_id: str) -> Optional[Dict[str, Any]]:
#     with get_conn() as conn:
#         row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
#         return dict(row) if row else None


# def list_videos() -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
#         return [dict(r) for r in rows]


# def save_frame_records(video_id: str, records: List[tuple]) -> None:
#     with get_conn() as conn:
#         conn.execute("DELETE FROM frame_metadata WHERE video_id = ?", (video_id,))
#         conn.executemany(
#             """
#             INSERT INTO frame_metadata (video_id, faiss_idx, timestamp_sec, thumbnail_name)
#             VALUES (?, ?, ?, ?)
#             """,
#             records,
#         )
#         conn.commit()


# def get_frame_records(video_id: str) -> List[Dict[str, Any]]:
#     with get_conn() as conn:
#         rows = conn.execute(
#             """
#             SELECT faiss_idx, timestamp_sec, thumbnail_name
#             FROM frame_metadata
#             WHERE video_id = ?
#             ORDER BY faiss_idx ASC
#             """,
#             (video_id,),
#         ).fetchall()
#         return [dict(r) for r in rows]