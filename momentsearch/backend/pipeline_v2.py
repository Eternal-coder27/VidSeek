import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, List, Tuple

import cv2
import faiss
import numpy as np
from PIL import Image

from config import DEFAULT_SAMPLING_FPS, EMBEDDING_DIM, FRAMES_DIR, INDEXES_DIR
import database as db
from embedding import embedding_service


# ============================================================
# IMAGE ENHANCEMENT & CLIP PREPROCESSING
# ============================================================

def enhance_frame(bgr: np.ndarray) -> np.ndarray:
    """Denoise -> CLAHE -> Sharpen for low-quality/noisy frames."""
    h, w = bgr.shape[:2]
    if w > 960:
        bgr = cv2.resize(bgr, (960, max(1, int(h * 960 / w))), interpolation=cv2.INTER_AREA)

    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    return cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)


def branch_clip_pad_square(rgb: np.ndarray, size: int = 224) -> Image.Image:
    """[1:1 Pad] branch: preserves aspect ratio with neutral gray padding."""
    h, w, _ = rgb.shape
    scale = size / max(h, w, 1)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return Image.fromarray(canvas)


# ============================================================
# V2 INGESTION (NO YOLO / NO OCR)
# ============================================================

def process_video_task_v2(
    video_id: str,
    video_path: Path,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
) -> None:
    raw_temp_dir = FRAMES_DIR / video_id / "_raw_v2"
    try:
        db.update_video_status(
            video_id,
            status="processing",
            progress=2.0,
            sampling_fps=sampling_fps,
            pipeline="v2",
            error_message=None,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = round(total_frames / native_fps, 2) if total_frames > 0 else 0.0
        cap.release()

        video_frames_dir = FRAMES_DIR / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        if raw_temp_dir.exists():
            shutil.rmtree(raw_temp_dir)
        raw_temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. FFmpeg coarse extraction at 2x sampling_fps for sharpest-frame window
        coarse_fps = max(1.0, sampling_fps * 2.0)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={coarse_fps}",
                str(raw_temp_dir / "raw_%05d.jpg"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        raw_files = sorted(raw_temp_dir.glob("raw_*.jpg"))
        if not raw_files:
            raise ValueError("FFmpeg extracted zero frames from video.")

        db.update_video_status(
            video_id,
            status="processing",
            progress=15.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
        )

        # 2. Temporal Sampling: Sharpest-Frame Window
        selected: List[Tuple[float, Path, float]] = []
        for i in range(0, len(raw_files), 2):
            window = raw_files[i:i + 2]
            best_path, best_sharpness, best_ts = window[0], -1.0, round(i / coarse_fps, 2)
            for j, fp in enumerate(window):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                sharp = float(
                    cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                )
                if sharp > best_sharpness:
                    best_sharpness = sharp
                    best_path = fp
                    best_ts = round((i + j) / coarse_fps, 2)
            selected.append((best_ts, best_path, best_sharpness))

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        records = []

        for idx, (ts, fp, sharpness) in enumerate(selected):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue

            # 3. Image Processing (Denoise -> CLAHE -> Sharpen)
            enhanced_bgr = enhance_frame(bgr)

            thumb_name = f"frame_v2_{idx:05d}.jpg"
            h, w = enhanced_bgr.shape[:2]
            scale = 360.0 / max(w, 1)
            preview = cv2.resize(enhanced_bgr, (360, max(1, int(h * scale))))
            cv2.imwrite(
                str(video_frames_dir / thumb_name),
                preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )

            # 4. [1:1 Pad] -> CLIP Embedding
            rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            sq_img = branch_clip_pad_square(rgb)
            vec = embedding_service.encode_image(sq_img)
            index.add(vec)

            # Empty objects/ocr_text while YOLO & OCR are disabled
            records.append((video_id, idx, ts, thumb_name, sharpness, json.dumps([]), ""))

            pct = min(99.0, round(15.0 + ((idx + 1) / len(selected)) * 84.0, 1))
            db.update_video_status(
                video_id,
                status="processing",
                progress=pct,
                duration_sec=duration_sec,
                fps=round(native_fps, 2),
                frame_count=idx + 1,
            )

        if not records:
            raise ValueError("No valid frames processed in V2 pipeline.")

        faiss.write_index(index, str(INDEXES_DIR / f"{video_id}_v2.faiss"))
        db.save_frame_records_v2(video_id, records)

        db.update_video_status(
            video_id,
            status="completed",
            progress=100.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
            frame_count=len(records),
            sampling_fps=sampling_fps,
            pipeline="v2",
            error_message=None,
        )
    except Exception as exc:
        db.update_video_status(
            video_id, status="failed", progress=0.0, error_message=str(exc)
        )
    finally:
        if raw_temp_dir.exists():
            shutil.rmtree(raw_temp_dir, ignore_errors=True)


# ============================================================
# V2 SEARCH, TEMPORAL GROUNDING & EVIDENCE VERIFICATION
# ============================================================

def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def search_video_index_v2(
    video_id: str,
    query: str,
    top_k: int = 5,
    merge_gap_sec: float = 2.0,
    sampling_fps: float = 1.0,
) -> List[dict[str, Any]]:
    index_path = INDEXES_DIR / f"{video_id}_v2.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            "V2 index not found. Re-process this video with the V2 pipeline first."
        )

    index = faiss.read_index(str(index_path))
    records = db.get_frame_records_v2(video_id)
    if not records or index.ntotal == 0:
        return []

    query_vec = embedding_service.encode_text(query)
    scores, indices = index.search(query_vec, index.ntotal)
    clip_map = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}

    scored_frames = []
    for rec in records:
        idx = int(rec["faiss_idx"])
        sim = max(0.0, clip_map.get(idx, 0.0))
        scored_frames.append(
            {
                "timestamp": float(rec["timestamp_sec"]),
                "score": sim,
                "thumbnail_name": rec["thumbnail_name"],
            }
        )

    # Temporal Grounding: smoothing + threshold grouping + boundary refinement
    raw = np.array([f["score"] for f in scored_frames])
    smooth = np.convolve(raw, np.ones(3) / 3.0, mode="same")
    thresh = max(float(np.mean(smooth) + 0.5 * np.std(smooth)), 0.20)

    clusters: List[List[dict[str, Any]]] = []
    current: List[dict[str, Any]] = []
    for i, frame in enumerate(scored_frames):
        if smooth[i] >= thresh:
            if current and (frame["timestamp"] - current[-1]["timestamp"] > merge_gap_sec):
                clusters.append(current)
                current = [frame]
            else:
                current.append(frame)
    if current:
        clusters.append(current)

    frame_step = 1.0 / max(sampling_fps, 0.1)
    proposals = []
    for cluster in clusters:
        start_t = max(0.0, cluster[0]["timestamp"] - frame_step * 0.5)
        end_t = cluster[-1]["timestamp"] + frame_step * 0.5
        peak = max(cluster, key=lambda x: x["score"])
        confidence = round(min(1.0, peak["score"] * 2.5), 3)

        kf_names = list(
            dict.fromkeys(
                [
                    cluster[0]["thumbnail_name"],
                    peak["thumbnail_name"],
                    cluster[-1]["thumbnail_name"],
                ]
            )
        )
        keyframes = [f"/api/videos/{video_id}/frames/{name}" for name in kf_names]
        review_flag = (
            "verified"
            if confidence >= 0.65
            else ("needs_review" if confidence >= 0.45 else "low_confidence")
        )

        proposals.append(
            {
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "peak_timestamp": round(peak["timestamp"], 2),
                "similarity_score": round(peak["score"], 4),
                "thumbnail_url": f"/api/videos/{video_id}/frames/{peak['thumbnail_name']}",
                "confidence": confidence,
                "ai_review_flag": review_flag,
                "detected_objects": [],
                "ocr_text": None,
                "keyframes": keyframes,
            }
        )

    # 1D Temporal NMS (IoU > 0.5)
    proposals.sort(key=lambda x: x["confidence"], reverse=True)
    kept: List[dict[str, Any]] = []
    for prop in proposals:
        if all(
            _temporal_iou((prop["start_time"], prop["end_time"]), (k["start_time"], k["end_time"])) <= 0.5
            for k in kept
        ):
            kept.append(prop)
        if len(kept) >= top_k:
            break

    for rank, item in enumerate(kept, start=1):
        item["rank"] = rank
    return kept

# import json
# from pathlib import Path
# import shutil
# import subprocess
# from typing import Any, Dict, List, Set, Tuple

# import cv2
# import faiss
# import numpy as np
# from PIL import Image
# import torch

# from config import DEFAULT_SAMPLING_FPS, EMBEDDING_DIM, FRAMES_DIR, INDEXES_DIR
# import database as db
# from embedding import embedding_service

# _yolo_model = None
# _ocr_reader = None


# def _get_multimodal_models():
#     global _yolo_model, _ocr_reader
#     if _yolo_model is None:
#         from ultralytics import YOLO
#         _yolo_model = YOLO("yolov8n.pt")
#     if _ocr_reader is None:
#         import easyocr
#         _ocr_reader = easyocr.Reader(["en"], gpu=torch.backends.mps.is_available())
#     return _yolo_model, _ocr_reader


# # ============================================================
# # IMAGE ENHANCEMENT & 3-BRANCH PREPROCESSING
# # ============================================================

# def enhance_frame(bgr: np.ndarray) -> np.ndarray:
#     """Denoise -> CLAHE -> Unsharp Mask for low-light/noisy CCTV frames."""
#     denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
#     lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l_eq = clahe.apply(l)
#     enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
#     blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
#     return cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)


# def branch_clip_pad_square(rgb: np.ndarray, size: int = 224) -> Image.Image:
#     """[1:1 Pad] branch: preserves aspect ratio with neutral gray padding."""
#     h, w, _ = rgb.shape
#     scale = size / max(h, w, 1)
#     nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
#     resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
#     canvas = np.full((size, size, 3), 114, dtype=np.uint8)
#     top, left = (size - nh) // 2, (size - nw) // 2
#     canvas[top:top + nh, left:left + nw] = resized
#     return Image.fromarray(canvas)


# def branch_ocr_gray_thresh(bgr: np.ndarray) -> np.ndarray:
#     """[Gray/Thresh] branch: adaptive thresholding for text/plates/signs."""
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)


# # ============================================================
# # V2 INGESTION TASK
# # ============================================================

# def process_video_task_v2(
#     video_id: str,
#     video_path: Path,
#     sampling_fps: float = DEFAULT_SAMPLING_FPS,
# ) -> None:
#     raw_temp_dir = FRAMES_DIR / video_id / "_raw_v2"
#     try:
#         db.update_video_status(
#             video_id,
#             status="processing",
#             progress=2.0,
#             sampling_fps=sampling_fps,
#             pipeline="v2",
#             error_message=None,
#         )

#         # Probe duration & native fps
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             raise ValueError("Unable to open video file.")
#         native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         duration_sec = round(total_frames / native_fps, 2) if total_frames > 0 else 0.0
#         cap.release()

#         video_frames_dir = FRAMES_DIR / video_id
#         video_frames_dir.mkdir(parents=True, exist_ok=True)
#         if raw_temp_dir.exists():
#             shutil.rmtree(raw_temp_dir)
#         raw_temp_dir.mkdir(parents=True, exist_ok=True)

#         # 1. FFmpeg coarse extraction at 2x target sampling rate for sharpest-frame selection
#         coarse_fps = max(1.0, sampling_fps * 2.0)
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={coarse_fps}", str(raw_temp_dir / "raw_%05d.jpg")],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             check=True,
#         )

#         raw_files = sorted(raw_temp_dir.glob("raw_*.jpg"))
#         if not raw_files:
#             raise ValueError("FFmpeg extracted zero frames from video.")

#         db.update_video_status(video_id, status="processing", progress=15.0, duration_sec=duration_sec, fps=round(native_fps, 2))

#         # 2. Temporal Sampling: Sharpest-Frame Window (window size = 2 frames)
#         selected: List[Tuple[float, Path, float]] = []
#         for i in range(0, len(raw_files), 2):
#             window = raw_files[i:i + 2]
#             best_path, best_sharpness, best_ts = window[0], -1.0, round(i / coarse_fps, 2)
#             for j, fp in enumerate(window):
#                 img = cv2.imread(str(fp))
#                 if img is None:
#                     continue
#                 sharp = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
#                 if sharp > best_sharpness:
#                     best_sharpness = sharp
#                     best_path = fp
#                     best_ts = round((i + j) / coarse_fps, 2)
#             selected.append((best_ts, best_path, best_sharpness))

#         yolo_model, ocr_reader = _get_multimodal_models()
#         index = faiss.IndexFlatIP(EMBEDDING_DIM)
#         records = []

#         for idx, (ts, fp, sharpness) in enumerate(selected):
#             bgr = cv2.imread(str(fp))
#             if bgr is None:
#                 continue

#             # 3. Image Processing (Denoise -> CLAHE -> Sharpen)
#             enhanced_bgr = enhance_frame(bgr)

#             thumb_name = f"frame_v2_{idx:05d}.jpg"
#             h, w = enhanced_bgr.shape[:2]
#             scale = 360.0 / max(w, 1)
#             preview = cv2.resize(enhanced_bgr, (360, max(1, int(h * scale))))
#             cv2.imwrite(str(video_frames_dir / thumb_name), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

#             # Branch 1: [1:1 Pad] -> CLIP
#             rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
#             sq_img = branch_clip_pad_square(rgb)
#             vec = embedding_service.encode_image(sq_img)
#             index.add(vec)

#             # Branch 2: [Native] -> YOLO
#             yolo_out = yolo_model(enhanced_bgr, verbose=False)[0]
#             objects = sorted({
#                 yolo_model.names[int(box.cls[0])]
#                 for box in yolo_out.boxes
#                 if float(box.conf[0]) >= 0.35
#             })

#             # Branch 3: [Gray/Thresh] -> OCR (every 2nd keyframe to keep M4 ingestion fast)
#             ocr_text = ""
#             if idx % 2 == 0:
#                 thresh_img = branch_ocr_gray_thresh(enhanced_bgr)
#                 hits = ocr_reader.readtext(thresh_img, detail=0)
#                 ocr_text = " ".join(hits).strip()

#             records.append((video_id, idx, ts, thumb_name, sharpness, json.dumps(objects), ocr_text))

#             if (idx + 1) % 3 == 0:
#                 pct = min(95.0, round(15.0 + ((idx + 1) / len(selected)) * 80.0, 1))
#                 db.update_video_status(
#                     video_id,
#                     status="processing",
#                     progress=pct,
#                     duration_sec=duration_sec,
#                     fps=round(native_fps, 2),
#                     frame_count=idx + 1,
#                 )

#         if not records:
#             raise ValueError("No valid frames processed in V2 pipeline.")

#         faiss.write_index(index, str(INDEXES_DIR / f"{video_id}_v2.faiss"))
#         db.save_frame_records_v2(video_id, records)

#         db.update_video_status(
#             video_id,
#             status="completed",
#             progress=100.0,
#             duration_sec=duration_sec,
#             fps=round(native_fps, 2),
#             frame_count=len(records),
#             sampling_fps=sampling_fps,
#             pipeline="v2",
#             error_message=None,
#         )
#     except Exception as exc:
#         db.update_video_status(video_id, status="failed", progress=0.0, error_message=str(exc))
#     finally:
#         if raw_temp_dir.exists():
#             shutil.rmtree(raw_temp_dir, ignore_errors=True)


# # ============================================================
# # V2 SEARCH, TEMPORAL GROUNDING & EVIDENCE VERIFICATION
# # ============================================================

# def _parse_query(query: str) -> Tuple[Set[str], List[str]]:
#     tokens = [t.strip(".,!?\"'()") for t in query.split()]
#     coco_vocab = {
#         "person", "man", "woman", "boy", "girl", "car", "truck", "bus", "motorcycle",
#         "bicycle", "backpack", "bag", "handbag", "suitcase", "laptop", "cell phone", "phone", "dog"
#     }
#     yolo_terms = {t.lower() for t in tokens if t.lower() in coco_vocab}
#     if yolo_terms.intersection({"man", "woman", "boy", "girl"}):
#         yolo_terms.add("person")
#     if "bag" in yolo_terms:
#         yolo_terms.update({"backpack", "handbag", "suitcase"})

#     ocr_terms = [
#         t.lower() for t in tokens
#         if len(t) >= 2 and (any(c.isdigit() for c in t) or any(c.isupper() for c in t[1:]) or t.isupper())
#     ]
#     return yolo_terms, ocr_terms


# def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
#     inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
#     union = max(a[1], b[1]) - min(a[0], b[0])
#     return inter / union if union > 0 else 0.0


# def search_video_index_v2(
#     video_id: str,
#     query: str,
#     top_k: int = 5,
#     merge_gap_sec: float = 2.0,
#     sampling_fps: float = 1.0,
# ) -> List[Dict[str, Any]]:
#     index_path = INDEXES_DIR / f"{video_id}_v2.faiss"
#     if not index_path.exists():
#         raise FileNotFoundError("V2 multimodal index not found. Re-process this video with the V2 pipeline first.")

#     index = faiss.read_index(str(index_path))
#     records = db.get_frame_records_v2(video_id)
#     if not records or index.ntotal == 0:
#         return []

#     # 1. Query Encoder (CLIP + Parser)
#     query_vec = embedding_service.encode_text(query)
#     yolo_terms, ocr_terms = _parse_query(query)

#     scores, indices = index.search(query_vec, index.ntotal)
#     clip_map = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}

#     # 2. Candidate Retrieval (Hybrid Vector + Metadata)
#     scored_frames = []
#     for rec in records:
#         idx = int(rec["faiss_idx"])
#         c_sim = max(0.0, clip_map.get(idx, 0.0))
#         objs = set(json.loads(rec["objects_json"] or "[]"))
#         ocr_str = rec["ocr_text"] or ""

#         yolo_hit = 1.0 if (yolo_terms and objs.intersection(yolo_terms)) else 0.0
#         ocr_hit = 1.0 if (ocr_terms and any(term in ocr_str.lower() for term in ocr_terms)) else 0.0

#         if yolo_terms or ocr_terms:
#             hybrid = 0.65 * c_sim + 0.20 * yolo_hit + 0.15 * ocr_hit
#         else:
#             hybrid = c_sim

#         scored_frames.append({
#             "timestamp": float(rec["timestamp_sec"]),
#             "hybrid": hybrid,
#             "clip": c_sim,
#             "thumbnail_name": rec["thumbnail_name"],
#             "objects": sorted(objs),
#             "ocr_text": ocr_str,
#         })

#     # 3. Temporal Grounding (Smoothing + Dynamic Boundary Refinement + IoU NMS)
#     raw = np.array([f["hybrid"] for f in scored_frames])
#     smooth = np.convolve(raw, np.ones(3) / 3.0, mode="same")
#     dynamic_thresh = max(float(np.mean(smooth) + 0.5 * np.std(smooth)), 0.20)

#     clusters: List[List[Dict[str, Any]]] = []
#     current: List[Dict[str, Any]] = []
#     for i, frame in enumerate(scored_frames):
#         if smooth[i] >= dynamic_thresh:
#             if current and (frame["timestamp"] - current[-1]["timestamp"] > merge_gap_sec):
#                 clusters.append(current)
#                 current = [frame]
#             else:
#                 current.append(frame)
#     if current:
#         clusters.append(current)

#     frame_step = 1.0 / max(sampling_fps, 0.1)
#     proposals = []
#     for cluster in clusters:
#         start_t = max(0.0, cluster[0]["timestamp"] - frame_step * 0.5)
#         end_t = cluster[-1]["timestamp"] + frame_step * 0.5
#         peak = max(cluster, key=lambda x: x["hybrid"])
#         confidence = round(min(1.0, peak["hybrid"] * 2.3), 3)

#         # 4. Evidence Verification (Keyframes + AI Review Flag)
#         kf_names = list(dict.fromkeys([
#             cluster[0]["thumbnail_name"],
#             peak["thumbnail_name"],
#             cluster[-1]["thumbnail_name"],
#         ]))
#         keyframes = [f"/api/videos/{video_id}/frames/{name}" for name in kf_names]
#         review_flag = "verified" if confidence >= 0.65 else ("needs_review" if confidence >= 0.45 else "low_confidence")

#         proposals.append({
#             "start_time": round(start_t, 2),
#             "end_time": round(end_t, 2),
#             "peak_timestamp": round(peak["timestamp"], 2),
#             "similarity_score": round(peak["hybrid"], 4),
#             "thumbnail_url": f"/api/videos/{video_id}/frames/{peak['thumbnail_name']}",
#             "confidence": confidence,
#             "ai_review_flag": review_flag,
#             "detected_objects": peak["objects"],
#             "ocr_text": peak["ocr_text"] or None,
#             "keyframes": keyframes,
#         })

#     proposals.sort(key=lambda x: x["confidence"], reverse=True)
#     kept: List[Dict[str, Any]] = []
#     for prop in proposals:
#         if all(_temporal_iou((prop["start_time"], prop["end_time"]), (k["start_time"], k["end_time"])) <= 0.5 for k in kept):
#             kept.append(prop)
#         if len(kept) >= top_k:
#             break

#     for rank, item in enumerate(kept, start=1):
#         item["rank"] = rank
#     return kept