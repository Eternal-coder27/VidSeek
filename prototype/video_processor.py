import cv2

from PIL import Image


def extract_frames(
    video_path: str,
    sample_fps: float = 1.0
):
    """
    Extract frames at a fixed temporal rate.

    Returns:

        [
            {
                "timestamp": float,
                "image": PIL.Image
            },
            ...
        ]
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: "
            f"{video_path}"
        )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if video_fps <= 0:

        cap.release()

        raise ValueError(
            "Invalid video FPS."
        )

    duration = (
        frame_count /
        video_fps
    )

    interval = (
        1.0 /
        sample_fps
    )

    frames = []

    timestamp = 0.0

    while timestamp < duration:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame
        )

        frames.append(
            {
                "timestamp": timestamp,
                "image": image
            }
        )

        timestamp += interval

    cap.release()

    print(
        f"Extracted {len(frames)} frames "
        f"from {duration:.2f}s video."
    )

    return frames