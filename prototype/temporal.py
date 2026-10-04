def calculate_iou(
    prediction,
    ground_truth
):
    """
    Calculate temporal Intersection over Union.

    prediction:
        [start, end]

    ground_truth:
        [start, end]
    """

    pred_start, pred_end = prediction

    gt_start, gt_end = ground_truth

    intersection_start = max(
        pred_start,
        gt_start
    )

    intersection_end = min(
        pred_end,
        gt_end
    )

    intersection = max(
        0.0,
        intersection_end -
        intersection_start
    )

    union_start = min(
        pred_start,
        gt_start
    )

    union_end = max(
        pred_end,
        gt_end
    )

    union = (
        union_end -
        union_start
    )

    if union <= 0:
        return 0.0

    return intersection / union


def frame_to_segment(
    timestamp: float,
    frame_duration: float = 1.0
):
    """
    Convert a sampled frame into a
    one-second temporal segment.
    """

    return [
        timestamp,
        timestamp + frame_duration
    ]


def group_results(
    results,
    max_gap=2.0
):
    """
    Group temporally close retrieved frames.

    Example:

        10s
        11s
        12s

    becomes:

        [10, 12]

    """

    if not results:
        return []

    # Sort by timestamp
    results = sorted(
        results,
        key=lambda x: x["timestamp"]
    )

    groups = []

    current_group = [
        results[0]
    ]

    for result in results[1:]:

        previous = (
            current_group[-1]
        )

        gap = (
            result["timestamp"]
            -
            previous["timestamp"]
        )

        if gap <= max_gap:

            current_group.append(
                result
            )

        else:

            groups.append(
                current_group
            )

            current_group = [
                result
            ]

    groups.append(
        current_group
    )

    return groups


def group_to_segment(
    group
):
    """
    Convert a temporal group into
    [start, end].
    """

    start = min(
        item["timestamp"]
        for item in group
    )

    end = max(
        item["timestamp"]
        for item in group
    ) + 1.0

    return [
        start,
        end
    ]


def best_temporal_segment(
    results,
    max_gap=2.0
):
    """
    Find the strongest temporal candidate.
    """

    if not results:
        return None

    groups = group_results(
        results,
        max_gap=max_gap
    )

    candidates = []

    for group in groups:

        segment = group_to_segment(
            group
        )

        # Average score of the group
        average_score = (
            sum(
                item["score"]
                for item in group
            )
            /
            len(group)
        )

        # Maximum score
        max_score = max(
            item["score"]
            for item in group
        )

        candidates.append(
            {
                "start": segment[0],
                "end": segment[1],
                "average_score": average_score,
                "max_score": max_score,
                "frame_count": len(group)
            }
        )

    # Strongest group
    candidates.sort(
        key=lambda x: x["average_score"],
        reverse=True
    )

    return candidates[0]