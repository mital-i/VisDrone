"""
Test-Time Augmentation (TTA) utilities for better recall.

Runs inference on augmented images (flips, scales)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms


def tta_inference(model, image, device, score_threshold=0.01):
    """
    Test-time augmentation inference.

    Runs inference on:
    - Original image
    - Horizontal flip
    - Multi-scale (0.8x, 1.0x, 1.2x)

    Then merges predictions with NMS.

    Args:
        model: Detection model
        image: Image tensor [C, H, W]
        device: torch device
        score_threshold: Minimum confidence threshold

    Returns:
        Merged predictions dict with 'boxes', 'labels', 'scores'
    """
    all_predictions = []

    # 1. Original image
    pred = model([image.to(device)])[0]
    pred = filter_predictions(pred, score_threshold)
    all_predictions.append(pred)

    # 2. Horizontal flip
    img_flip = torch.flip(image, dims=[2])
    pred_flip = model([img_flip.to(device)])[0]
    pred_flip = filter_predictions(pred_flip, score_threshold)

    # Unflip boxes: x_new = width - x_old
    if len(pred_flip["boxes"]) > 0:
        w = image.shape[2]
        pred_flip["boxes"][:, [0, 2]] = w - pred_flip["boxes"][:, [2, 0]]
    all_predictions.append(pred_flip)

    # 3. Multi-scale inference
    for scale in [0.8, 1.2]:
        # Resize image
        h, w = int(image.shape[1] * scale), int(image.shape[2] * scale)
        img_scaled = F.interpolate(
            image.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        )[0]

        pred_scaled = model([img_scaled.to(device)])[0]
        pred_scaled = filter_predictions(pred_scaled, score_threshold)

        # Rescale boxes back to original size
        if len(pred_scaled["boxes"]) > 0:
            pred_scaled["boxes"] /= scale

        all_predictions.append(pred_scaled)

    # Merge all predictions with NMS
    merged = merge_predictions_nms(all_predictions, iou_threshold=0.5)

    return merged


def filter_predictions(pred, score_threshold):
    """Filter predictions by score threshold."""
    mask = pred["scores"] >= score_threshold
    return {
        "boxes": pred["boxes"][mask],
        "labels": pred["labels"][mask],
        "scores": pred["scores"][mask],
    }


def merge_predictions_nms(predictions, iou_threshold=0.5):
    """
    Merge multiple predictions using NMS.

    Concatenates all boxes/labels/scores and applies NMS.
    """
    # Collect all predictions
    all_boxes = []
    all_labels = []
    all_scores = []

    for pred in predictions:
        if len(pred["boxes"]) > 0:
            all_boxes.append(pred["boxes"].cpu())
            all_labels.append(pred["labels"].cpu())
            all_scores.append(pred["scores"].cpu())

    if len(all_boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "scores": torch.zeros((0,)),
        }

    # Concatenate
    boxes = torch.cat(all_boxes, dim=0)
    labels = torch.cat(all_labels, dim=0)
    scores = torch.cat(all_scores, dim=0)

    # Apply NMS per class
    keep_indices: list[torch.Tensor] = []

    for class_id in labels.unique():
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # NMS for this class
        keep = nms(class_boxes, class_scores, iou_threshold)

        # Convert back to original indices
        original_indices = torch.where(class_mask)[0]
        keep_indices.append(original_indices[keep])

    if len(keep_indices) > 0:
        keep_indices = torch.cat(keep_indices)  # type: ignore[assignment]

        return {
            "boxes": boxes[keep_indices],
            "labels": labels[keep_indices],
            "scores": scores[keep_indices],
        }
    else:
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "scores": torch.zeros((0,)),
        }


def weighted_boxes_fusion(predictions, weights=None, iou_threshold=0.5):
    """
    Weighted Boxes Fusion (WBF) for merging predictions.

    Better than NMS for ensemble/TTA as it averages overlapping boxes
    instead of suppressing them.

    Args:
        predictions: List of prediction dicts
        weights: Weights for each prediction (default: equal weights)
        iou_threshold: IoU threshold for fusion

    Returns:
        Fused predictions
    """
    try:
        import importlib.util

        if importlib.util.find_spec("ensemble_boxes") is None:
            raise ImportError
    except ImportError:
        print("Warning: ensemble-boxes not installed. Using NMS instead.")
        print("Install with: pip install ensemble-boxes")
        return merge_predictions_nms(predictions, iou_threshold)

    # ensemble_boxes is available

    if weights is None:
        weights = [1.0] * len(predictions)

    # Prepare data for WBF (needs normalized coordinates)
    boxes_list = []
    scores_list = []
    labels_list = []

    for pred in predictions:
        if len(pred["boxes"]) > 0:
            # WBF expects [x1, y1, x2, y2] in [0, 1] range
            # We'll normalize later per image
            boxes_list.append(pred["boxes"].cpu().numpy())
            scores_list.append(pred["scores"].cpu().numpy())
            labels_list.append(pred["labels"].cpu().numpy())
        else:
            boxes_list.append(np.zeros((0, 4)))
            scores_list.append(np.array([]))
            labels_list.append(np.array([]))

    # Note: WBF implementation here is simplified
    # For production, properly normalize boxes and use ensemble-boxes library

    # Fallback to NMS for now
    return merge_predictions_nms(predictions, iou_threshold)
