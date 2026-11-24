import os

# Must execute before importing torch or faiss on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from config import (
    ALLOWED_EXTENSIONS,
    DEFAULT_SAMPLING_FPS,
    FRAMES_DIR,
    MAX_UPLOAD_BYTES,
    VIDEOS_DIR,
)
import database as db
from embedding import embedding_service
from pipeline_v2 import process_video_task_v2, search_video_index_v2
from pipeline_v3 import process_video_task_v3, search_video_index_v3
from schemas import ProcessRequest, SearchRequest, SearchResponse, VideoMetadata
from search_index import search_video_index
from video_processor import process_video_task

VALID_PIPELINES = {"v1", "v2", "v3"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    embedding_service.load()
    yield


app = FastAPI(title="MomentSearch API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_uuid(video_id: str) -> str:
    try:
        return str(uuid.UUID(video_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")


def _get_processor_for_pipeline(pipeline: str):
    if pipeline == "v3":
        return process_video_task_v3
    if pipeline == "v2":
        return process_video_task_v2
    return process_video_task


@app.post("/api/videos/upload", response_model=VideoMetadata)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pipeline: str = Form("v1"),
):
    if pipeline not in VALID_PIPELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline '{pipeline}'. Allowed: v1, v2, v3.",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    video_id = str(uuid.uuid4())
    dest_path = VIDEOS_DIR / f"{video_id}{ext}"

    bytes_written = 0
    try:
        with open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    out_file.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
    task_fn = _get_processor_for_pipeline(pipeline)
    background_tasks.add_task(task_fn, video_id, dest_path, DEFAULT_SAMPLING_FPS)

    record = db.get_video(video_id)
    assert record is not None
    return record


@app.get("/api/videos", response_model=List[VideoMetadata])
def get_videos():
    return db.list_videos()


@app.get("/api/videos/{video_id}", response_model=VideoMetadata)
def get_video_metadata(video_id: str):
    vid = validate_uuid(video_id)
    record = db.get_video(vid)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    return record


@app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
def start_video_processing(
    video_id: str,
    background_tasks: BackgroundTasks,
    payload: ProcessRequest = ProcessRequest(),
):
    vid = validate_uuid(video_id)
    record = db.get_video(vid)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    if payload.pipeline not in VALID_PIPELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline '{payload.pipeline}'. Allowed: v1, v2, v3.",
        )

    video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing from storage.")

    db.update_video_status(
        vid,
        status="processing",
        progress=0.0,
        pipeline=payload.pipeline,
        error_message=None,
    )
    task_fn = _get_processor_for_pipeline(payload.pipeline)
    background_tasks.add_task(task_fn, vid, video_path, payload.sampling_fps)

    updated = db.get_video(vid)
    assert updated is not None
    return updated


@app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
def get_video_status(video_id: str):
    vid = validate_uuid(video_id)
    record = db.get_video(vid)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    return record


@app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
async def search_video(video_id: str, payload: SearchRequest):
    vid = validate_uuid(video_id)
    record = db.get_video(vid)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    if record["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Video is not ready for search (status: {record['status']}).",
        )

    active_pipeline = payload.pipeline or record.get("pipeline") or "v1"
    if active_pipeline not in VALID_PIPELINES:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline '{active_pipeline}'.")

    sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)

    if active_pipeline == "v3":
        try:
            matches = await search_video_index_v3(
                video_id=vid,
                query=payload.query,
                top_k=payload.top_k,
                rerank_top_k=min(payload.top_k, 4),
                merge_gap_sec=payload.merge_gap_sec,
                sampling_fps=sampling_fps,
                llm_provider=payload.llm_provider,
                llm_model=payload.llm_model,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    elif active_pipeline == "v2":
        try:
            matches = search_video_index_v2(
                video_id=vid,
                query=payload.query,
                top_k=payload.top_k,
                merge_gap_sec=payload.merge_gap_sec,
                sampling_fps=sampling_fps,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    else:
        query_vec = embedding_service.encode_text(payload.query)
        matches = search_video_index(
            video_id=vid,
            query_vec=query_vec,
            top_k=payload.top_k,
            merge_gap_sec=payload.merge_gap_sec,
            sampling_fps=sampling_fps,
        )

    return SearchResponse(
        video_id=vid,
        query=payload.query,
        pipeline=active_pipeline,
        disclaimer="Retrieval scores indicate visual/multimodal similarity, not proof that an incident occurred. Human review is required.",
        results=matches,
    )


@app.get("/api/videos/{video_id}/frames/{frame_name}")
def get_frame_thumbnail(video_id: str, frame_name: str):
    vid = validate_uuid(video_id)
    safe_name = Path(frame_name).name
    if safe_name != frame_name or not safe_name.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="Invalid frame filename.")

    frame_path = FRAMES_DIR / vid / safe_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request):
    vid = validate_uuid(video_id)
    record = db.get_video(vid)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    media_type = "video/quicktime" if record["stored_ext"] == ".mov" else "video/mp4"

    if not range_header:
        return FileResponse(video_path, media_type=media_type)

    range_val = range_header.strip().replace("bytes=", "")
    start_str, end_str = range_val.split("-")
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else file_size - 1
    end = min(end, file_size - 1)

    if start >= file_size or start > end:
        raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

    chunk_length = end - start + 1

    def iter_file(path: Path, offset: int, length: int, chunk_size: int = 1024 * 1024):
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_file(video_path, start, chunk_length),
        status_code=206,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_length),
        },
        media_type=media_type,
    )

# import os

# # Must execute before importing torch or faiss on macOS
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import List
# import uuid

# from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
# from pipeline_v2 import process_video_task_v2, search_video_index_v2
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, StreamingResponse

# from config import (
#     ALLOWED_EXTENSIONS,
#     DEFAULT_SAMPLING_FPS,
#     FRAMES_DIR,
#     MAX_UPLOAD_BYTES,
#     VIDEOS_DIR,
# )
# import database as db
# from embedding import embedding_service
# from schemas import ProcessRequest, SearchRequest, SearchResponse, VideoMetadata
# from search_index import search_video_index
# from video_processor import process_video_task
# from pipeline_v3 import process_video_task_v3, search_video_index_v3

# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     db.init_db()
#     embedding_service.load()
#     yield


# app = FastAPI(title="MomentSearch API", version="1.0.0", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def validate_uuid(video_id: str) -> str:
#     try:
#         return str(uuid.UUID(video_id))
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid video ID format.")
    

# @app.post("/api/videos/upload", response_model=VideoMetadata)
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     pipeline: str = Form("v1"),
# ):
#     if pipeline not in {"v1", "v2"}:
#         raise HTTPException(status_code=400, detail="Pipeline must be 'v1' or 'v2'.")
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
#         )

#     video_id = str(uuid.uuid4())
#     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

#     bytes_written = 0
#     try:
#         with open(dest_path, "wb") as out_file:
#             while chunk := await file.read(1024 * 1024):
#                 bytes_written += len(chunk)
#                 if bytes_written > MAX_UPLOAD_BYTES:
#                     out_file.close()
#                     dest_path.unlink(missing_ok=True)
#                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
#                 out_file.write(chunk)
#     except HTTPException:
#         raise
#     except Exception as exc:
#         dest_path.unlink(missing_ok=True)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

#     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext, pipeline=pipeline)
#     task_fn = process_video_task_v2 if pipeline == "v2" else process_video_task
#     background_tasks.add_task(task_fn, video_id, dest_path, DEFAULT_SAMPLING_FPS)

#     record = db.get_video(video_id)
#     assert record is not None
#     return record


# # @app.post("/api/videos/upload", response_model=VideoMetadata)
# # async def upload_video(
# #     background_tasks: BackgroundTasks,
# #     file: UploadFile = File(...),
# # ):
# #     if not file.filename:
# #         raise HTTPException(status_code=400, detail="Missing filename.")

# #     ext = Path(file.filename).suffix.lower()
# #     if ext not in ALLOWED_EXTENSIONS:
# #         raise HTTPException(
# #             status_code=400,
# #             detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
# #         )

# #     video_id = str(uuid.uuid4())
# #     dest_path = VIDEOS_DIR / f"{video_id}{ext}"

# #     bytes_written = 0
# #     try:
# #         with open(dest_path, "wb") as out_file:
# #             while chunk := await file.read(1024 * 1024):
# #                 bytes_written += len(chunk)
# #                 if bytes_written > MAX_UPLOAD_BYTES:
# #                     out_file.close()
# #                     dest_path.unlink(missing_ok=True)
# #                     raise HTTPException(status_code=413, detail="File exceeds 500 MB upload limit.")
# #                 out_file.write(chunk)
# #     except HTTPException:
# #         raise
# #     except Exception as exc:
# #         dest_path.unlink(missing_ok=True)
# #         raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

# #     db.create_video(video_id=video_id, filename=file.filename, stored_ext=ext)
# #     background_tasks.add_task(process_video_task, video_id, dest_path, DEFAULT_SAMPLING_FPS)

# #     record = db.get_video(video_id)
# #     assert record is not None
# #     return record


# @app.get("/api/videos", response_model=List[VideoMetadata])
# def get_videos():
#     return db.list_videos()


# @app.get("/api/videos/{video_id}", response_model=VideoMetadata)
# def get_video_metadata(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record

# @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# def start_video_processing(
#     video_id: str,
#     background_tasks: BackgroundTasks,
#     payload: ProcessRequest = ProcessRequest(),
# ):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file missing from storage.")

#     db.update_video_status(vid, status="processing", progress=0.0, pipeline=payload.pipeline, error_message=None)
#     task_fn = process_video_task_v2 if payload.pipeline == "v2" else process_video_task
#     background_tasks.add_task(task_fn, vid, video_path, payload.sampling_fps)

#     updated = db.get_video(vid)
#     assert updated is not None
#     return updated


# # @app.post("/api/videos/{video_id}/process", response_model=VideoMetadata)
# # def start_video_processing(
# #     video_id: str,
# #     background_tasks: BackgroundTasks,
# #     payload: ProcessRequest = ProcessRequest(),
# # ):
# #     vid = validate_uuid(video_id)
# #     record = db.get_video(vid)
# #     if not record:
# #         raise HTTPException(status_code=404, detail="Video not found.")

# #     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
# #     if not video_path.exists():
# #         raise HTTPException(status_code=404, detail="Video file missing from storage.")

# #     db.update_video_status(vid, status="processing", progress=0.0, error_message=None)
# #     background_tasks.add_task(process_video_task, vid, video_path, payload.sampling_fps)

# #     updated = db.get_video(vid)
# #     assert updated is not None
# #     return updated


# @app.get("/api/videos/{video_id}/status", response_model=VideoMetadata)
# def get_video_status(video_id: str):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     return record

# @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# def search_video(video_id: str, payload: SearchRequest):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")
#     if record["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Video is not ready for search (status: {record['status']}).",
#         )

#     active_pipeline = payload.pipeline or record.get("pipeline") or "v1"
#     sampling_fps = float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS)

#     if active_pipeline == "v2":
#         try:
#             matches = search_video_index_v2(
#                 video_id=vid,
#                 query=payload.query,
#                 top_k=payload.top_k,
#                 merge_gap_sec=payload.merge_gap_sec,
#                 sampling_fps=sampling_fps,
#             )
#         except FileNotFoundError as exc:
#             raise HTTPException(status_code=409, detail=str(exc))
#     else:
#         query_vec = embedding_service.encode_text(payload.query)
#         matches = search_video_index(
#             video_id=vid,
#             query_vec=query_vec,
#             top_k=payload.top_k,
#             merge_gap_sec=payload.merge_gap_sec,
#             sampling_fps=sampling_fps,
#         )

#     return SearchResponse(
#         video_id=vid,
#         query=payload.query,
#         pipeline=active_pipeline,
#         disclaimer="Retrieval scores indicate multimodal AI similarity, not proof that an incident occurred. Human review is required.",
#         results=matches,
#     )


# # @app.post("/api/videos/{video_id}/search", response_model=SearchResponse)
# # def search_video(video_id: str, payload: SearchRequest):
# #     vid = validate_uuid(video_id)
# #     record = db.get_video(vid)
# #     if not record:
# #         raise HTTPException(status_code=404, detail="Video not found.")
# #     if record["status"] != "completed":
# #         raise HTTPException(
# #             status_code=409,
# #             detail=f"Video is not ready for search (status: {record['status']}).",
# #         )

# #     query_vec = embedding_service.encode_text(payload.query)
# #     matches = search_video_index(
# #         video_id=vid,
# #         query_vec=query_vec,
# #         top_k=payload.top_k,
# #         merge_gap_sec=payload.merge_gap_sec,
# #         sampling_fps=float(record.get("sampling_fps") or DEFAULT_SAMPLING_FPS),
# #     )

# #     return SearchResponse(
# #         video_id=vid,
# #         query=payload.query,
# #         disclaimer="Retrieval similarity scores indicate CLIP visual-text similarity, not proof that an incident occurred. Human review is required.",
# #         results=matches,
# #     )


# @app.get("/api/videos/{video_id}/frames/{frame_name}")
# def get_frame_thumbnail(video_id: str, frame_name: str):
#     vid = validate_uuid(video_id)
#     safe_name = Path(frame_name).name
#     if safe_name != frame_name or not safe_name.endswith(".jpg"):
#         raise HTTPException(status_code=400, detail="Invalid frame filename.")

#     frame_path = FRAMES_DIR / vid / safe_name
#     if not frame_path.exists():
#         raise HTTPException(status_code=404, detail="Frame thumbnail not found.")
#     return FileResponse(frame_path, media_type="image/jpeg")


# @app.get("/api/videos/{video_id}/stream")
# def stream_video(video_id: str, request: Request):
#     vid = validate_uuid(video_id)
#     record = db.get_video(vid)
#     if not record:
#         raise HTTPException(status_code=404, detail="Video not found.")

#     video_path = VIDEOS_DIR / f"{vid}{record['stored_ext']}"
#     if not video_path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found.")

#     file_size = video_path.stat().st_size
#     range_header = request.headers.get("range")
#     media_type = "video/quicktime" if record["stored_ext"] == ".mov" else "video/mp4"

#     if not range_header:
#         return FileResponse(video_path, media_type=media_type)

#     range_val = range_header.strip().replace("bytes=", "")
#     start_str, end_str = range_val.split("-")
#     start = int(start_str) if start_str else 0
#     end = int(end_str) if end_str else file_size - 1
#     end = min(end, file_size - 1)

#     if start >= file_size or start > end:
#         raise HTTPException(status_code=416, detail="Requested range not satisfiable.")

#     chunk_length = end - start + 1

#     def iter_file(path: Path, offset: int, length: int, chunk_size: int = 1024 * 1024):
#         with open(path, "rb") as f:
#             f.seek(offset)
#             remaining = length
#             while remaining > 0:
#                 data = f.read(min(chunk_size, remaining))
#                 if not data:
#                     break
#                 remaining -= len(data)
#                 yield data

#     return StreamingResponse(
#         iter_file(video_path, start, chunk_length),
#         status_code=206,
#         headers={
#             "Content-Range": f"bytes {start}-{end}/{file_size}",
#             "Accept-Ranges": "bytes",
#             "Content-Length": str(chunk_length),
#         },
#         media_type=media_type,
#     )