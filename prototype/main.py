import argparse

from index_video import build_index

from search import VideoSearcher

from temporal import best_temporal_segment


def format_timestamp(
    seconds
):

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    args = parser.parse_args()

    # Build index
    build_index(
        args.video,
        "indexes/manual.index",
        "indexes/manual.json"
    )

    searcher = VideoSearcher(
        "indexes/manual.index",
        "indexes/manual.json"
    )

    print("\n")

    print("=" * 60)

    print(
        "ChronoVision Prototype"
    )

    print("=" * 60)

    while True:

        query = input(
            "\nEnter query "
            "(or 'exit'):\n> "
        )

        if query.lower() == "exit":
            break

        results = searcher.search(
            query,
            top_k=10
        )

        print("\nTop frames:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank:02d}. "
                f"{format_timestamp(result['timestamp'])}"
                f" | "
                f"score={result['score']:.4f}"
            )

        segment = (
            best_temporal_segment(
                results
            )
        )

        if segment:

            print("\nPredicted moment:")

            print(
                f"["
                f"{format_timestamp(segment['start'])}"
                f" → "
                f"{format_timestamp(segment['end'])}"
                f"]"
            )


if __name__ == "__main__":
    main()