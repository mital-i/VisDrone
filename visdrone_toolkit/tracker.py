"""DeepSORT tracker integration for VisDrone detection pipeline."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSORTTracker:
    """Thin wrapper around :class:`deep_sort_realtime.DeepSort`.

    Parameters
    ----------
    max_age : int
        Maximum number of frames to keep a lost track alive.
    n_init : int
        Number of consecutive detections before a track is confirmed.
    max_iou_distance : float
        Maximum IoU distance for data association.
    max_cosine_distance : float
        Maximum cosine distance for appearance matching.
    nn_budget : int | None
        Feature gallery size per track (``None`` = unlimited).
    embedder : str
        Appearance feature extractor backend.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int | None = 100,
        embedder: str = "mobilenet",
    ) -> None:
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> list[dict]:
        """Feed one frame of detections and return confirmed tracks.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (used by the embedder for appearance features).
        boxes : np.ndarray
            ``(N, 4)`` array of ``[x1, y1, x2, y2]`` bounding boxes.
        scores : np.ndarray
            ``(N,)`` confidence scores.
        class_ids : np.ndarray
            ``(N,)`` integer class indices.

        Returns
        -------
        list[dict]
            Each dict has keys ``track_id``, ``ltrb`` (x1,y1,x2,y2),
            ``class_id``, ``confidence``, and ``center``.
        """
        # deep-sort-realtime expects: list of ([left, top, w, h], confidence, class)
        raw_detections = []
        for box, score, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            raw_detections.append(
                ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(score), int(cls))
            )

        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        # Collect confirmed tracks
        results: list[dict] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            # Skip coasting tracks (no matched detection this frame) to
            # avoid Kalman-predicted boxes that drift and enlarge.
            if track.time_since_update > 0:
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            class_id = track.det_class if track.det_class is not None else -1
            confidence = track.det_conf if track.det_conf is not None else 0.0
            center = (int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2))

            results.append(
                {
                    "track_id": track_id,
                    "ltrb": [int(v) for v in ltrb],
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "center": center,
                }
            )

        return results

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        class_names: Sequence[str],
        line_width: int = 2,
    ) -> np.ndarray:
        """Draw tracked bounding boxes and IDs on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to annotate (will be copied).
        tracks : list[dict]
            Output of :meth:`update`.
        class_names : Sequence[str]
            Ordered class name list for label rendering.
        line_width : int
            Bounding-box thickness.

        Returns
        -------
        np.ndarray
            Annotated frame copy.
        """
        frame = frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = t["ltrb"]
            track_id = t["track_id"]
            cls_id = t["class_id"]

            color = _color_for_id(cls_id)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

            # Label: "ID: class conf"
            if 0 <= cls_id < len(class_names):
                cls_name = class_names[cls_id]
            else:
                cls_name = f"cls_{cls_id}"
            conf = t.get("confidence", 0.0)
            label = f"{track_id}: {cls_name} {conf:.2f}"

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 4),
                (x1 + tw, y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def reset(self) -> None:
        """Clear tracker state (e.g. between sequences)."""
        self._tracker.delete_all_tracks()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_PALETTE = [
    (255, 0, 0),       # pedestrian - blue
    (255, 128, 0),     # people - light blue
    (0, 255, 0),       # bicycle - green
    (0, 255, 255),     # car - yellow
    (0, 128, 255),     # van - orange
    (0, 0, 255),       # truck - red
    (255, 0, 255),     # tricycle - magenta
    (128, 0, 255),     # awning-tricycle - purple
    (255, 255, 0),     # bus - cyan
    (128, 255, 0),     # motor - lime
]


def _color_for_id(class_id: int) -> tuple[int, int, int]:
    if 0 <= class_id < len(_PALETTE):
        return _PALETTE[class_id]
    return (
        (class_id * 47) % 256,
        (class_id * 97) % 256,
        (class_id * 157) % 256,
    )
