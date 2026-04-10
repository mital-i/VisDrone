"""Evaluate DeepSORT tracking on VisDrone MOT-val sequences."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import os

import cv2
import motmetrics as mm
import numpy as np

# Disable network access (FIPS compliance)
os.environ["YOLO_OFFLINE"] = "1"

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. pip install ultralytics")
    sys.exit(1)

# Add parent so visdrone_toolkit is importable when running from VisDrone/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visdrone_toolkit.tracker import DeepSORTTracker


# VisDrone MOT ignores category 0 (ignored-regions) and 11 (others)
IGNORED_CATEGORIES = {0, 11}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DeepSORT on VisDrone MOT-val",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    p.add_argument(
        "--data-dir",
        required=True,
        help="Root of MOT split (e.g. data/VisDrone2019-MOT-val). "
        "Must contain sequences/ and annotations/ sub-dirs.",
    )
    p.add_argument(
        "--weights",
        required=True,
        help="Path to YOLOv8 weights (.pt)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save per-sequence prediction files.",
    )

    # Detection
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    p.add_argument("--device", default="cpu", help="Device (cuda:0, cpu, …)")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    p.add_argument(
        "--use-visdrone-classes",
        action="store_true",
        help="Use VisDrone class names (for fine-tuned models)",
    )

    # Tracker
    p.add_argument("--track-max-age", type=int, default=30)
    p.add_argument("--track-n-init", type=int, default=3)
    p.add_argument("--track-max-cosine-distance", type=float, default=0.2)

    # Evaluation
    p.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Run only on these sequence names (default: all)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames per sequence (for quick tests)",
    )

    return p.parse_args()


# ------------------------------------------------------------------
# Ground-truth loader
# ------------------------------------------------------------------

def load_gt(annotation_path: Path) -> dict[int, list[dict]]:
    """Load VisDrone MOT ground-truth file.

    Returns mapping  frame_id → list of gt objects, each a dict with
    keys: target_id, bbox (x1,y1,x2,y2), category, score.
    """
    gt: dict[int, list[dict]] = {}
    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            target_id = int(parts[1])
            left, top, w, h = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            score = int(parts[6])
            category = int(parts[7])

            # Skip ignored annotations
            if score == 0 or category in IGNORED_CATEGORIES:
                continue

            gt.setdefault(frame_id, []).append(
                {
                    "target_id": target_id,
                    "bbox": [left, top, left + w, top + h],
                    "category": category,
                }
            )
    return gt


# ------------------------------------------------------------------
# IoU distance (avoids deprecated np.asfarray in motmetrics)
# ------------------------------------------------------------------

def _iou_distance_matrix(
    gt_boxes: np.ndarray,
    hyp_boxes: np.ndarray,
    max_iou: float = 0.5,
) -> np.ndarray:
    """Compute cost matrix of (1 - IoU) between gt and hypothesis boxes.

    Entries where IoU < (1 - max_iou) are set to NaN (= "no match possible"),
    matching the convention used by ``motmetrics``.

    Parameters
    ----------
    gt_boxes, hyp_boxes : (N,4) and (M,4) arrays in [x1,y1,x2,y2] format.
    max_iou : float – distance threshold; pairs with distance > max_iou are NaN.
    """
    ng = len(gt_boxes)
    nh = len(hyp_boxes)
    if ng == 0 or nh == 0:
        return np.full((ng, nh), np.nan)

    gt = np.asarray(gt_boxes, dtype=float)
    hyp = np.asarray(hyp_boxes, dtype=float)

    # Intersection
    x1 = np.maximum(gt[:, None, 0], hyp[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], hyp[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], hyp[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], hyp[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_hyp = (hyp[:, 2] - hyp[:, 0]) * (hyp[:, 3] - hyp[:, 1])
    union = area_gt[:, None] + area_hyp[None, :] - inter

    iou = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - iou
    dist[dist > max_iou] = np.nan
    return dist


# ------------------------------------------------------------------
# Per-sequence evaluation
# ------------------------------------------------------------------

def evaluate_sequence(
    model: YOLO,
    tracker: DeepSORTTracker,
    seq_dir: Path,
    gt: dict[int, list[dict]],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    half: bool,
    max_frames: int | None,
) -> mm.MOTAccumulator:
    """Run tracker on one sequence, accumulate MOT events."""
    acc = mm.MOTAccumulator(auto_id=True)
    tracker.reset()

    frame_files = sorted(
        f
        for f in seq_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if max_frames:
        frame_files = frame_files[:max_frames]

    for idx, frame_path in enumerate(frame_files):
        frame_id = idx + 1  # VisDrone frames are 1-indexed
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # --- Detection ---
        results = model.predict(
            frame, conf=conf, iou=iou, imgsz=imgsz,
            device=device, half=half, verbose=False,
        )[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        # --- Tracking ---
        tracks = tracker.update(frame, boxes, scores, classes)

        # --- Build GT / hypothesis arrays for this frame ---
        gt_frame = gt.get(frame_id, [])
        gt_ids = [g["target_id"] for g in gt_frame]
        gt_boxes = np.array([g["bbox"] for g in gt_frame]) if gt_frame else np.empty((0, 4))

        hyp_ids = [t["track_id"] for t in tracks]
        hyp_boxes = np.array([t["ltrb"] for t in tracks], dtype=float) if tracks else np.empty((0, 4))

        # Compute IoU distance matrix (1 - IoU), NaN where IoU < threshold
        dist = _iou_distance_matrix(gt_boxes, hyp_boxes, max_iou=0.5)

        acc.update(gt_ids, hyp_ids, dist)

    return acc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    seq_root = data_dir / "sequences"
    ann_root = data_dir / "annotations"

    if not seq_root.is_dir() or not ann_root.is_dir():
        print(f"Error: expected sequences/ and annotations/ under {data_dir}")
        return 1

    # Discover sequences
    if args.sequences:
        seq_names = args.sequences
    else:
        seq_names = sorted(d.name for d in seq_root.iterdir() if d.is_dir())

    if not seq_names:
        print("No sequences found.")
        return 1

    # Load model
    print(f"\n{'=' * 60}")
    print("DeepSORT MOT Evaluation on VisDrone")
    print(f"{'=' * 60}")
    print(f"Weights      : {args.weights}")
    print(f"Data dir     : {data_dir}")
    print(f"Sequences    : {len(seq_names)}")
    print(f"Conf / IoU   : {args.conf} / {args.iou}")
    print(f"Image size   : {args.imgsz}")
    print(f"Device       : {args.device}")
    print(f"Tracker      : DeepSORT (max_age={args.track_max_age}, "
          f"n_init={args.track_n_init}, cos_dist={args.track_max_cosine_distance})")
    print(f"{'=' * 60}\n")

    model = YOLO(args.weights)
    tracker = DeepSORTTracker(
        max_age=args.track_max_age,
        n_init=args.track_n_init,
        max_cosine_distance=args.track_max_cosine_distance,
    )

    # Optional output dir for predictions
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each sequence
    accumulators: list[mm.MOTAccumulator] = []
    names: list[str] = []

    for seq_name in seq_names:
        seq_dir = seq_root / seq_name
        ann_file = ann_root / f"{seq_name}.txt"
        if not seq_dir.is_dir():
            print(f"  [skip] sequence dir not found: {seq_dir}")
            continue
        if not ann_file.is_file():
            print(f"  [skip] annotation not found: {ann_file}")
            continue

        gt = load_gt(ann_file)
        n_frames = len(list(seq_dir.iterdir()))
        print(f"  {seq_name}  ({n_frames} frames, {sum(len(v) for v in gt.values())} gt objects) ...", end=" ", flush=True)

        t0 = time.time()
        acc = evaluate_sequence(
            model=model,
            tracker=tracker,
            seq_dir=seq_dir,
            gt=gt,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            max_frames=args.max_frames,
        )
        elapsed = time.time() - t0

        accumulators.append(acc)
        names.append(seq_name)
        print(f"done ({elapsed:.1f}s)")

    if not accumulators:
        print("No sequences evaluated.")
        return 1

    # Compute summary
    print(f"\n{'=' * 60}")
    print("MOT Metrics Summary")
    print(f"{'=' * 60}\n")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        names=names,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    print(f"\n{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
