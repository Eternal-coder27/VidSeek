from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent        # .../momentsearch/backend
MOMENTSEARCH_DIR = BACKEND_DIR.parent                # .../momentsearch
PROJECT_ROOT = MOMENTSEARCH_DIR.parent               # .../workspace_root

STORAGE_DIR = BACKEND_DIR / "storage"
VIDEOS_DIR = STORAGE_DIR / "videos"
FRAMES_DIR = STORAGE_DIR / "frames"
DB_PATH = STORAGE_DIR / "momentsearch.db"
INDEXES_DIR = PROJECT_ROOT / "indexes"               # Uses your root indexes/ folder

for folder in (VIDEOS_DIR, FRAMES_DIR, INDEXES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_ID = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
DEFAULT_SAMPLING_FPS = 1.0

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}