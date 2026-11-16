from pathlib import Path
import cv2
import faiss
from PIL import Image
from config import DEFAULT_SAMPLING_FPS, EMBEDDING_DIM, FRAMES_DIR
import database as db
from embedding import embedding_service
from search_index import save_video_index


def process_video_task(
    video_id: str,
    video_path: Path,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
) -> None:
    try:
        db.update_video_status(
            video_id,
            status="processing",
            progress=1.0,
            sampling_fps=sampling_fps,
            error_message=None,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open video. File may be corrupted or unsupported.")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = round(total_frames / native_fps, 2) if total_frames > 0 else 0.0
        step = max(1, int(round(native_fps / sampling_fps)))

        video_frames_dir = FRAMES_DIR / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        records = []
        frame_idx = 0
        sampled_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp_sec = round(frame_idx / native_fps, 2)
                thumb_name = f"frame_{sampled_count:05d}.jpg"
                thumb_path = video_frames_dir / thumb_name

                # Save lightweight thumbnail preview
                h, w = frame.shape[:2]
                scale = 360.0 / max(w, 1)
                resized = cv2.resize(frame, (360, max(1, int(h * scale))))
                cv2.imwrite(str(thumb_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

                # Generate normalized CLIP embedding
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vec = embedding_service.encode_image(Image.fromarray(rgb))
                index.add(vec)

                records.append((video_id, sampled_count, timestamp_sec, thumb_name))
                sampled_count += 1

                if total_frames > 0 and sampled_count % 5 == 0:
                    pct = min(95.0, round((frame_idx / total_frames) * 95.0, 1))
                    db.update_video_status(
                        video_id,
                        status="processing",
                        progress=pct,
                        duration_sec=duration_sec,
                        fps=round(native_fps, 2),
                        frame_count=sampled_count,
                    )

            frame_idx += 1

        cap.release()

        if sampled_count == 0:
            raise ValueError("No readable frames could be extracted from this video.")

        save_video_index(video_id, index)
        db.save_frame_records(video_id, records)

        db.update_video_status(
            video_id,
            status="completed",
            progress=100.0,
            duration_sec=duration_sec,
            fps=round(native_fps, 2),
            frame_count=sampled_count,
            sampling_fps=sampling_fps,
            error_message=None,
        )
    except Exception as exc:
        db.update_video_status(
            video_id,
            status="failed",
            progress=0.0,
            error_message=str(exc),
        )