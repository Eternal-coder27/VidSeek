from pathlib import Path

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"

ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"

INDEX_DIR = PROJECT_ROOT / "indexes"

ANNOTATION_FILE = (
    ANNOTATION_DIR / "charades-sta-test-small.csv"
)


# ============================================================
# CLIP
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

# Number of frames processed simultaneously.
BATCH_SIZE = 32


# ============================================================
# Video sampling
# ============================================================

SAMPLE_FPS = 1.0


# ============================================================
# Retrieval
# ============================================================

TOP_K = 10


# ============================================================
# Temporal grouping
# ============================================================

# Frames within this many seconds can belong
# to the same event.
TEMPORAL_WINDOW = 2.0


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"

elif torch.cuda.is_available():
    DEVICE = "cuda"

else:
    DEVICE = "cpu"


# ============================================================
# Create directories
# ============================================================

VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR = (
    PROJECT_ROOT / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)