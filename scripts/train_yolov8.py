#!/usr/bin/env python3
"""
YOLOv8 Training Script for VisDrone Dataset.

Based on the training workflow from train-yolov8-object-detection-on-custom-dataset.ipynb.

Usage:
    python scripts/train_yolov8.py --data data/visdrone.yaml --epochs 100
    python scripts/train_yolov8.py --model yolov8s.pt --epochs 50 --imgsz 1280
    python scripts/train_yolov8.py --resume runs/detect/train/weights/last.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Disable telemetry and network access before importing ultralytics (FIPS compliance)
os.environ["YOLO_SETTINGS_SYNC"] = "False"
os.environ["YOLO_OFFLINE"] = "1"

try:
    from ultralytics import YOLO
    import ultralytics
except ImportError:
    print("Error: ultralytics package not found.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on VisDrone dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--model", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", default="data/visdrone.yaml", help="Dataset YAML")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="", help="Device: cuda, cpu, 0, 1")

    # Optimizer
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--cache", action="store_true", help="Cache images")

    # Hardware optimization
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision (default: True)")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    # Output
    parser.add_argument("--project", default="runs/detect", help="Project directory")
    parser.add_argument("--name", default="visdrone", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate plots")

    # Resume
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")

    return parser.parse_args()


def check_gpu():
    """Display GPU information."""
    import torch

    print("\n" + "=" * 60)
    if torch.cuda.is_available():
        print(f"CUDA: Yes | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: No (training on CPU)")
    print(f"PyTorch: {torch.__version__} | Ultralytics: {ultralytics.__version__}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    check_gpu()

    # Resume training
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return 0

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Determine AMP setting
    use_amp = args.amp and not getattr(args, 'no_amp', False)

    # Train
    print(f"\nTraining on: {args.data}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Image size: {args.imgsz} | AMP: {use_amp}\n")

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=args.plots,
        device=args.device if args.device else None,
        amp=use_amp,
    )

    # Validate
    print("\nRunning validation...")
    weights = Path(args.project) / args.name / "weights" / "best.pt"
    if weights.exists():
        model = YOLO(str(weights))
        model.val(data=args.data, imgsz=args.imgsz)

    print(f"\nDone! Best weights: {weights}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
