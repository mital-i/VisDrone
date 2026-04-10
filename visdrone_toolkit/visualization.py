"""
Visualization utilities for VisDrone dataset and predictions.

Provides functions to visualize ground truth annotations and model predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure, SubFigure
from PIL import Image

# VisDrone class colors (RGB)
CLASS_COLORS = {
    0: (128, 128, 128),  # ignored-regions - gray
    1: (255, 0, 0),  # pedestrian - red
    2: (255, 128, 0),  # people - orange
    3: (255, 255, 0),  # bicycle - yellow
    4: (0, 255, 0),  # car - green
    5: (0, 255, 128),  # van - light green
    6: (0, 255, 255),  # truck - cyan
    7: (0, 128, 255),  # tricycle - light blue
    8: (0, 0, 255),  # awning-tricycle - blue
    9: (128, 0, 255),  # bus - purple
    10: (255, 0, 255),  # motor - magenta
    11: (255, 0, 128),  # others - pink
}

CLASS_NAMES = [
    "ignored-regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]


def visualize_annotations(
    image: np.ndarray | Image.Image,
    boxes: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    title: str = "Ground Truth",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure:
    """
    Visualize ground truth annotations.

    Args:
        image: Input image (numpy array or PIL Image)
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes
        labels: (N,) array of class labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the figure
        ax: Optional matplotlib axis to plot on

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Create figure
    fig: Figure | SubFigure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Draw boxes
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Get color for class
        color = np.array(CLASS_COLORS.get(int(label), (255, 255, 255))) / 255.0

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Add label text
        class_name = CLASS_NAMES[int(label)] if 0 <= label < len(CLASS_NAMES) else f"class_{label}"
        ax.text(
            x1,
            y1 - 5,
            class_name,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.7},
            fontsize=8,
            color="white",
            weight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def visualize_predictions(
    image: np.ndarray | Image.Image,
    boxes: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    score_threshold: float = 0.5,
    title: str = "Predictions",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | Path | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> Figure | SubFigure:
    """
    Visualize model predictions.

    Args:
        image: Input image (numpy array or PIL Image)
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes
        labels: (N,) array of class labels
        scores: (N,) array of confidence scores
        score_threshold: Minimum score to display
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the figure
        ax: Optional matplotlib axis to plot on

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # Create figure
    fig: Figure | SubFigure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(image)
    ax.set_title(f"{title} (threshold={score_threshold:.2f})", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Get color for class
        color = np.array(CLASS_COLORS.get(int(label), (255, 255, 255))) / 255.0

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Add label text with score
        class_name = CLASS_NAMES[int(label)] if 0 <= label < len(CLASS_NAMES) else f"class_{label}"
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1,
            y1 - 5,
            label_text,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.7},
            fontsize=8,
            color="white",
            weight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def visualize_comparison(
    image: np.ndarray | Image.Image,
    gt_boxes: np.ndarray | torch.Tensor,
    gt_labels: np.ndarray | torch.Tensor,
    pred_boxes: np.ndarray | torch.Tensor,
    pred_labels: np.ndarray | torch.Tensor,
    pred_scores: np.ndarray | torch.Tensor,
    score_threshold: float = 0.5,
    figsize: tuple[int, int] = (20, 8),
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """
    Visualize ground truth and predictions side by side.

    Args:
        image: Input image
        gt_boxes: Ground truth boxes
        gt_labels: Ground truth labels
        pred_boxes: Predicted boxes
        pred_labels: Predicted labels
        pred_scores: Prediction scores
        score_threshold: Minimum score to display
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Visualize ground truth
    visualize_annotations(image, gt_boxes, gt_labels, title="Ground Truth", show=False, ax=ax1)

    # Visualize predictions
    visualize_predictions(
        image,
        pred_boxes,
        pred_labels,
        pred_scores,
        score_threshold=score_threshold,
        title="Predictions",
        show=False,
        ax=ax2,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    metrics: dict[str, list[float]] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """
    Plot training curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        metrics: Dict of metric names to lists of values (optional)
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    n_plots = 1 + (1 if metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot metrics
    if metrics:
        ax = axes[1]
        for metric_name, values in metrics.items():
            ax.plot(epochs[: len(values)], values, "-o", label=metric_name, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Validation Metrics", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    if show:
        plt.show()

    return fig


def create_detection_grid(
    images: list[np.ndarray | Image.Image],
    predictions: list[dict[str, torch.Tensor]],
    score_threshold: float = 0.5,
    grid_size: tuple[int, int] = (2, 2),
    figsize: tuple[int, int] = (16, 16),
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """
    Create a grid of detection results.

    Args:
        images: List of images
        predictions: List of prediction dicts
        score_threshold: Score threshold for display
        grid_size: (rows, cols) for grid
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, (ax, image, pred) in enumerate(zip(axes.flat, images, predictions)):
        if idx >= len(images):
            ax.axis("off")
            continue

        visualize_predictions(
            image,
            pred["boxes"],
            pred["labels"],
            pred["scores"],
            score_threshold=score_threshold,
            title=f"Image {idx + 1}",
            show=False,
            ax=ax,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Detection grid saved to {save_path}")

    if show:
        plt.show()

    return fig
