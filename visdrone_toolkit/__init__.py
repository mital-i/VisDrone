"""VisDrone Toolkit - Modern PyTorch-based toolkit for VisDrone dataset.

A comprehensive toolkit for working with the VisDrone dataset, featuring:
- Native PyTorch Dataset class
- Multiple annotation format converters (COCO, YOLO)
- Visualization utilities
- DeepSORT multi-object tracking
- Training scripts for modern object detection models

"""

__version__ = "2.0.0"
__author__ = "Saumya Kumaar Saksena"
__license__ = "Apache-2.0"

from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.tracker import DeepSORTTracker
from visdrone_toolkit.utils import VISDRONE_CLASSES, collate_fn, get_model
from visdrone_toolkit.visualization import visualize_annotations, visualize_predictions

__all__ = [
    "VisDroneDataset",
    "DeepSORTTracker",
    "VISDRONE_CLASSES",
    "get_model",
    "collate_fn",
    "visualize_annotations",
    "visualize_predictions",
]
