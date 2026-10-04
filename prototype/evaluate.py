import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import hashlib
import json
from pathlib import Path


from config import (
    ANNOTATION_FILE,
    INDEX_DIR,
    VIDEO_DIR,
    TOP_K,
    EVALUATION_DIR
)

from clip_model import CLIPEncoder

from index_video import build_index

from search import VideoSearcher

from temporal import (
    calculate_iou,
    best_temporal_segment
)


# ============================================================
# Find video
# ============================================================

def find_video(
    video_id: str
) -> Path:

    extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    ]

    for extension in extensions:

        path = (
            VIDEO_DIR /
            f"{video_id}{extension}"
        )

        if path.exists():
            return path

    for path in VIDEO_DIR.rglob("*"):

        if (
            path.is_file()
            and path.stem == video_id
        ):

            return path

    raise FileNotFoundError(
        f"Video not found: {video_id}"
    )


# ============================================================
# Index filenames
# ============================================================

def get_index_paths(
    video_id
):

    safe_id = hashlib.md5(
        video_id.encode()
    ).hexdigest()[:12]

    index_path = (
        INDEX_DIR /
        f"{safe_id}.index"
    )

    metadata_path = (
        INDEX_DIR /
        f"{safe_id}.json"
    )

    return (
        index_path,
        metadata_path
    )


# ============================================================
# Evaluate query
# ============================================================

def evaluate_query(
    searcher,
    query,
    ground_truth
):

    results = searcher.search(
        query,
        top_k=TOP_K
    )

    prediction = (
        best_temporal_segment(
            results
        )
    )

    if prediction is None:

        return {
            "prediction": None,
            "iou": 0.0
        }

    predicted_segment = [
        prediction["start"],
        prediction["end"]
    ]

    iou = calculate_iou(
        predicted_segment,
        ground_truth
    )

    return {
        "prediction": predicted_segment,
        "iou": iou,
        "score": prediction[
            "average_score"
        ]
    }


# ============================================================
# Save JSON
# ============================================================

def save_json(
    data,
    filename
):

    path = (
        EVALUATION_DIR /
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print(
        f"Saved: {path}"
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    results
):

    path = (
        EVALUATION_DIR /
        "baseline_results.csv"
    )

    fields = [
        "video",
        "query",
        "ground_truth_start",
        "ground_truth_end",
        "predicted_start",
        "predicted_end",
        "iou",
        "similarity_score"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"Saved: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "ChronoVision Temporal "
        "Grounding Evaluation"
    )

    print("=" * 70)

    # ========================================================
    # Load CLIP ONCE
    # ========================================================

    encoder = CLIPEncoder()

    # ========================================================
    # Read CSV
    # ========================================================

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        annotations = list(
            reader
        )

    print(
        f"Queries found: "
        f"{len(annotations)}"
    )

    # ========================================================
    # Group queries
    # ========================================================

    videos = {}

    for row in annotations:

        video_id = (
            row["video"].strip()
        )

        videos.setdefault(
            video_id,
            []
        )

        videos[video_id].append(
            row
        )

    print(
        f"Videos found: "
        f"{len(videos)}"
    )

    # ========================================================
    # Metrics
    # ========================================================

    total_queries = 0

    correct_03 = 0
    correct_05 = 0
    correct_07 = 0

    all_ious = []

    detailed_results = []

    # ========================================================
    # Process videos
    # ========================================================

    for video_number, (
        video_id,
        queries
    ) in enumerate(
        videos.items(),
        start=1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"[{video_number}/"
            f"{len(videos)}] "
            f"Video: {video_id}"
        )

        try:

            video_path = find_video(
                video_id
            )

        except FileNotFoundError as error:

            print(
                f"WARNING: {error}"
            )

            continue

        index_path, metadata_path = (
            get_index_paths(
                video_id
            )
        )

        # ====================================================
        # Build index
        # ====================================================

        if not index_path.exists():

            build_index(
                str(video_path),
                str(index_path),
                str(metadata_path),
                encoder
            )

        else:

            print(
                "Using existing FAISS index."
            )

        # ====================================================
        # Searcher
        # ====================================================

        searcher = VideoSearcher(
            index_path,
            metadata_path,
            encoder
        )

        # ====================================================
        # Queries
        # ====================================================

        for row in queries:

            query = (
                row["query"].strip()
            )

            ground_truth = [
                float(row["start"]),
                float(row["end"])
            ]

            result = evaluate_query(
                searcher,
                query,
                ground_truth
            )

            iou = result["iou"]

            prediction = (
                result["prediction"]
            )

            similarity_score = result.get(
                "score",
                None
            )

            total_queries += 1

            all_ious.append(
                iou
            )

            # --------------------------------------------
            # Recall / IoU thresholds
            # --------------------------------------------

            if iou >= 0.3:
                correct_03 += 1

            if iou >= 0.5:
                correct_05 += 1

            if iou >= 0.7:
                correct_07 += 1

            # --------------------------------------------
            # Save detailed result
            # --------------------------------------------

            if prediction:

                predicted_start = (
                    prediction[0]
                )

                predicted_end = (
                    prediction[1]
                )

            else:

                predicted_start = None
                predicted_end = None

            detailed_results.append(
                {
                    "video": video_id,
                    "query": query,
                    "ground_truth_start": (
                        ground_truth[0]
                    ),
                    "ground_truth_end": (
                        ground_truth[1]
                    ),
                    "predicted_start": (
                        predicted_start
                    ),
                    "predicted_end": (
                        predicted_end
                    ),
                    "iou": iou,
                    "similarity_score": (
                        similarity_score
                    )
                }
            )

            print(
                f"\nQuery: {query}"
            )

            print(
                f"GT: "
                f"[{ground_truth[0]:.2f}, "
                f"{ground_truth[1]:.2f}]"
            )

            print(
                f"Prediction: "
                f"{prediction}"
            )

            print(
                f"IoU: {iou:.4f}"
            )

    # ========================================================
    # No results
    # ========================================================

    if total_queries == 0:

        print(
            "\nNo queries evaluated."
        )

        return

    # ========================================================
    # Metrics
    # ========================================================

    recall_03 = (
        correct_03 /
        total_queries
    ) * 100

    recall_05 = (
        correct_05 /
        total_queries
    ) * 100

    recall_07 = (
        correct_07 /
        total_queries
    ) * 100

    mean_iou = (
        sum(all_ious) /
        len(all_ious)
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "model": (
            "CLIP ViT-B/32"
        ),
        "retrieval": (
            "FAISS IndexFlatIP"
        ),
        "embedding_dimension": 512,
        "sampling_fps": 1.0,
        "top_k": TOP_K,
        "queries_evaluated": (
            total_queries
        ),
        "videos_evaluated": (
            len(videos)
        ),
        "recall_at_1_iou_0.3": (
            recall_03
        ),
        "recall_at_1_iou_0.5": (
            recall_05
        ),
        "recall_at_1_iou_0.7": (
            recall_07
        ),
        "mean_iou": mean_iou
    }

    # ========================================================
    # Save everything
    # ========================================================

    save_json(
        summary,
        "baseline_summary.json"
    )

    save_json(
        detailed_results,
        "baseline_results.json"
    )

    save_csv(
        detailed_results
    )

    # ========================================================
    # Terminal summary
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "FINAL BASELINE RESULTS"
    )

    print("=" * 70)

    print(
        f"Queries evaluated : "
        f"{total_queries}"
    )

    print(
        f"R@1 IoU@0.3       : "
        f"{recall_03:.2f}%"
    )

    print(
        f"R@1 IoU@0.5       : "
        f"{recall_05:.2f}%"
    )

    print(
        f"R@1 IoU@0.7       : "
        f"{recall_07:.2f}%"
    )

    print(
        f"Mean IoU          : "
        f"{mean_iou:.4f}"
    )

    print("=" * 70)

    print(
        "\nAll evaluation results saved "
        "inside the evaluation/ directory."
    )


if __name__ == "__main__":
    main()