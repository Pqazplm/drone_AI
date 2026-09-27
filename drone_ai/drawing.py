"""
Drawing functions for overlaying bounding boxes and labels on BGR frames.
"""

import cv2

from drone_ai.tracker import Track


def draw_detections(
    frame_bgr,
    tracks: list[Track],
    labels: list[str],
    frame_w: int,
    frame_h: int,
):
    """
    Draw every currently-alive track (fresh or coasting on persistence).

    Args:
        frame_bgr: BGR image (numpy array) to draw on.
        tracks: List of Track objects from the tracker.
        labels: List of class name strings.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        The modified frame_bgr (for convenience).
    """
    for t in tracks:
        x1, y1, x2, y2 = t.box
        x1 = int(max(0, x1) * frame_w)
        y1 = int(max(0, y1) * frame_h)
        x2 = int(min(1.0, x2) * frame_w)
        y2 = int(min(1.0, y2) * frame_h)

        label = (
            labels[t.class_id]
            if 0 <= t.class_id < len(labels)
            else str(t.class_id)
        )
        # misses>0 means this box is coasting (not re-confirmed this cycle) --
        # a dimmer color makes that visible to the pilot instead of hiding it.
        color = (0, 255, 0) if t.misses == 0 else (0, 200, 255)
        text = f"#{t.track_id} {label} {t.score:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame_bgr,
            (x1, max(0, y1 - th - 6)),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame_bgr,
            text,
            (x1 + 2, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame_bgr
