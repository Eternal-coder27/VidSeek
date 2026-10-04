import json

import faiss

from clip_model import CLIPEncoder


class VideoSearcher:

    def __init__(
        self,
        index_path,
        metadata_path,
        encoder: CLIPEncoder
    ):

        # Reuse the existing CLIP model
        self.encoder = encoder

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )

        self.timestamps = (
            metadata["timestamps"]
        )

    def search(
        self,
        query,
        top_k=10
    ):

        query_embedding = (
            self.encoder.encode_text(
                query
            )
        )

        scores, ids = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, index_id in zip(
            scores[0],
            ids[0]
        ):

            if index_id == -1:
                continue

            results.append(
                {
                    "timestamp": (
                        self.timestamps[
                            index_id
                        ]
                    ),
                    "score": float(
                        score
                    ),
                    "index_id": int(
                        index_id
                    )
                }
            )

        return results