import json

import faiss
import numpy as np

from tqdm import tqdm

from clip_model import CLIPEncoder
from video_processor import extract_frames

from config import (
    BATCH_SIZE,
    SAMPLE_FPS
)


def build_index(
    video_path: str,
    index_path: str,
    metadata_path: str,
    encoder: CLIPEncoder
):
    """
    Build a FAISS index for a video.

    The CLIP encoder is passed in so that
    the model is loaded only once.
    """

    print(
        f"\nIndexing: {video_path}"
    )

    frames = extract_frames(
        video_path,
        sample_fps=SAMPLE_FPS
    )

    if not frames:

        raise ValueError(
            "No frames extracted."
        )

    timestamps = [
        item["timestamp"]
        for item in frames
    ]

    images = [
        item["image"]
        for item in frames
    ]

    embeddings = []

    print(
        "Generating CLIP embeddings..."
    )

    for start in tqdm(
        range(
            0,
            len(images),
            BATCH_SIZE
        )
    ):

        end = min(
            start + BATCH_SIZE,
            len(images)
        )

        batch = images[start:end]

        batch_embeddings = (
            encoder.encode_images(
                batch
            )
        )

        embeddings.append(
            batch_embeddings
        )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    ).astype(np.float32)

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    dimension = embeddings.shape[1]

    # Normalized embeddings + Inner Product
    # = cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal}"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    metadata = {
        "video_path": str(video_path),
        "timestamps": timestamps,
        "sample_fps": SAMPLE_FPS
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print("Index saved.")

    return index