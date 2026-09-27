"""
Lightweight IoU-based tracker to reduce bounding-box flicker.

WHY FLICKER HAPPENS
-------------------
With ~600-750ms inference and inference only every 3rd captured frame, a
detection is a rare, noisy event. A person standing still can drop below
conf_thresh on one inference cycle and reappear on the next just from YOLO's
own frame-to-frame score jitter. Drawing "only what the current inference
cycle says" makes that jitter visible as boxes popping in and out.

WHAT FIXES IT
-------------
1. Higher conf_thresh + higher NMS IoU — raises the bar so borderline boxes
   never appear at all.
2. Track persistence — a track is only dropped after `max_misses` consecutive
   missed inference cycles, not missed display frames.
3. EMA smoothing — when a track gets a fresh match, its displayed box moves
   toward the new box gradually (`smooth_alpha`) rather than snapping.
4. Minimal IoU-based greedy matching between inference cycles — lets us say
   "this new detection is the same object as that track" so we know what to
   smooth toward and what to keep alive.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    box: np.ndarray        # smoothed xyxy, normalized [0,1], shape (4,)
    score: float
    class_id: int
    misses: int = 0         # consecutive INFERENCE CYCLES with no matching detection
    age: int = 0            # inference cycles this track has been alive/matched


class SimpleIoUTracker:
    """
    Greedy IoU-matching tracker across inference cycles (NOT display frames).

    Call update() once per inference cycle with the fresh (filtered, NMS'd)
    detections; call get_display_tracks() every display frame to get what to
    draw (including tracks currently "coasting" on persistence).
    """

    def __init__(self, iou_match_thresh: float, smooth_alpha: float, max_misses: int):
        self.iou_match_thresh = iou_match_thresh
        self.smooth_alpha = smooth_alpha
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Vectorized pairwise IoU. boxes_a: [N,4], boxes_b: [M,4], both xyxy."""
        if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
            return np.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

        ax1, ay1 = boxes_a[:, 0:1], boxes_a[:, 1:2]
        ax2, ay2 = boxes_a[:, 2:3], boxes_a[:, 3:4]
        bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        xx1 = np.maximum(ax1, bx1)
        yy1 = np.maximum(ay1, by1)
        xx2 = np.minimum(ax2, bx2)
        yy2 = np.minimum(ay2, by2)

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
        area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def update(
        self,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        det_class_ids: np.ndarray,
    ):
        """
        Process a new set of detections and update track states.

        Args:
            det_boxes: [N, 4] xyxy boxes in normalized [0,1] coordinates.
            det_scores: [N] confidence scores.
            det_class_ids: [N] class id integers.
        """
        track_boxes = (
            np.stack([t.box for t in self.tracks])
            if self.tracks
            else np.empty((0, 4), dtype=np.float32)
        )
        iou = self._iou_matrix(track_boxes, det_boxes)

        # Greedy match: highest-IoU pairs first, same class only.
        candidates = []
        for ti in range(iou.shape[0]):
            for di in range(iou.shape[1]):
                if (
                    self.tracks[ti].class_id == det_class_ids[di]
                    and iou[ti, di] >= self.iou_match_thresh
                ):
                    candidates.append((iou[ti, di], ti, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            t = self.tracks[ti]
            a = self.smooth_alpha
            t.box = a * det_boxes[di] + (1.0 - a) * t.box  # EMA toward new box
            t.score = float(det_scores[di])
            t.misses = 0
            t.age += 1

        # Unmatched existing tracks: age them; keep coasting until max_misses
        surviving = []
        for ti, t in enumerate(self.tracks):
            if ti in matched_tracks:
                surviving.append(t)
                continue
            t.misses += 1
            if t.misses <= self.max_misses:
                surviving.append(t)  # keep drawing at last known (smoothed) box
        self.tracks = surviving

        # Unmatched detections: brand-new tracks
        for di in range(det_boxes.shape[0]):
            if di in matched_dets:
                continue
            self.tracks.append(Track(
                track_id=self._next_id,
                box=det_boxes[di].copy(),
                score=float(det_scores[di]),
                class_id=int(det_class_ids[di]),
                misses=0,
                age=1,
            ))
            self._next_id += 1

    def get_display_tracks(self) -> list[Track]:
        """Return all currently alive tracks (fresh or coasting)."""
        return self.tracks
