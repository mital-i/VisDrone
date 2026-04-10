"""YOLOv8 Video Inference Script for VisDrone Sequences.

Supports inference on:
- Video files (.mp4, .avi, .mov, etc.)
- Frame sequences (directories with numbered images)
- Single images

Optionally applies DeepSORT multi-object tracking (--track flag).
Outputs video with bounding boxes, class labels, and track IDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure visdrone_toolkit is importable when running from VisDrone/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import cv2
import numpy as np

# Disable network access (FIPS compliance)
os.environ["YOLO_OFFLINE"] = "1"

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


# VisDrone class names (excluding ignored-regions)
VISDRONE_CLASSES = [
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
]

# Color palette for visualization (BGR format)
CLASS_COLORS = [
    (255, 0, 0),      # pedestrian - blue
    (255, 128, 0),    # people - light blue
    (0, 255, 0),      # bicycle - green
    (0, 255, 255),    # car - yellow
    (0, 128, 255),    # van - orange
    (0, 0, 255),      # truck - red
    (255, 0, 255),    # tricycle - magenta
    (128, 0, 255),    # awning-tricycle - purple
    (255, 255, 0),    # bus - cyan
    (128, 255, 0),    # motor - lime
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 inference on VisDrone video sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Run on a frame sequence directory
    python yolov8_video_inference.py \\
        --input data/VisDrone2019-VID-val/sequences/uav0000086_00000_v \\
        --weights runs/detect/visdrone/weights/best.pt \\
        --output-dir results

    # Run on a video file with custom threshold
    python yolov8_video_inference.py \\
        --input video.mp4 \\
        --weights runs/detect/train/weights/best.pt \\
        --conf 0.3 \\
        --output-dir results

    # Process all sequences in a directory
    python yolov8_video_inference.py \\
        --input data/VisDrone2019-VID-val/sequences \\
        --weights runs/detect/visdrone/weights/best.pt \\
        --batch-sequences \\
        --output-dir results
    """,
    )

    # Model
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to YOLOv8 weights (.pt)",
    )

    # Input/Output
    parser.add_argument(
        "--input",
        required=True,
        help="Path to video file, frame sequence directory, or parent directory with --batch-sequences",
    )
    parser.add_argument(
        "--output-dir",
        default="yolov8_outputs",
        help="Output directory for result videos (default: yolov8_outputs)",
    )

    # Inference parameters
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size (default: 1280)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cuda:0, cpu, etc.). Default: cpu",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half-precision inference (GPU only)",
    )

    # Processing options
    parser.add_argument(
        "--batch-sequences",
        action="store_true",
        help="Process all sequence subdirectories in the input path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS for frame sequences (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: all)",
    )

    # Visualization options
    parser.add_argument(
        "--use-visdrone-classes",
        action="store_true",
        help="Use VisDrone class names (for fine-tuned models)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width (default: 2)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in window during processing",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results to text files",
    )

    # Tracking options
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable DeepSORT multi-object tracking",
    )
    parser.add_argument(
        "--track-max-age",
        type=int,
        default=30,
        help="Max frames to keep lost tracks (default: 30)",
    )
    parser.add_argument(
        "--track-n-init",
        type=int,
        default=3,
        help="Detections needed to confirm a track (default: 3)",
    )
    parser.add_argument(
        "--track-max-cosine-distance",
        type=float,
        default=0.2,
        help="Max cosine distance for appearance matching (default: 0.2)",
    )

    return parser.parse_args()


def get_frame_files(sequence_dir: Path) -> list[Path]:
    """Get sorted list of frame files from a sequence directory."""
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    frames = []
    for ext in extensions:
        frames.extend(sequence_dir.glob(f"*{ext}"))
        frames.extend(sequence_dir.glob(f"*{ext.upper()}"))
    return sorted(frames, key=lambda x: x.stem)


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    line_width: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    frame = frame.copy()
    
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls)
        
        # Get class name and color
        if cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"class_{cls_id}"
        
        if cls_id < len(CLASS_COLORS):
            color = CLASS_COLORS[cls_id]
        else:
            # Generate color from class id
            color = (
                (cls_id * 47) % 256,
                (cls_id * 97) % 256,
                (cls_id * 157) % 256,
            )
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        
        # Prepare label
        label = f"{class_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 4),
            (x1 + text_width, y1),
            color,
            -1,
        )
        
        # Draw label text
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


def process_frame_sequence(
    model: YOLO,
    sequence_dir: Path,
    output_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    fps: int,
    class_names: list[str],
    line_width: int,
    show: bool,
    save_txt: bool,
    device: str = "cpu",
    half: bool = False,
    max_frames: int | None = None,
    tracker=None,
) -> dict:
    """Process a frame sequence directory and output video."""
    frames = get_frame_files(sequence_dir)
    
    if len(frames) == 0:
        print(f"  Warning: No frames found in {sequence_dir}")
        return {"frames": 0, "detections": 0}
    
    if max_frames:
        frames = frames[:max_frames]
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print(f"  Error: Could not read {frames[0]}")
        return {"frames": 0, "detections": 0}
    
    height, width = first_frame.shape[:2]
    
    # Create two video writers - one with annotations, one without
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Annotated video (with bounding boxes)
    annotated_path = output_path.parent / f"{output_path.stem}_annotated{output_path.suffix}"
    writer_annotated = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
    
    # Clean video (without bounding boxes)
    clean_path = output_path.parent / f"{output_path.stem}_clean{output_path.suffix}"
    writer_clean = cv2.VideoWriter(str(clean_path), fourcc, fps, (width, height))
    
    # Create text output directory if needed
    txt_output_dir = None
    if save_txt:
        txt_output_dir = output_path.parent / f"{output_path.stem}_labels"
        txt_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_detections = 0
    
    # Reset tracker state for this sequence
    if tracker is not None:
        tracker.reset()
    
    print(f"  Processing {len(frames)} frames...")
    
    for i, frame_path in enumerate(frames):
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Run inference
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )[0]
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        total_detections += len(boxes)
        
        # Draw with tracker or plain detections
        if tracker is not None:
            tracks = tracker.update(frame, boxes, scores, classes)
            annotated_frame = tracker.draw_tracks(
                frame, tracks, class_names, line_width,
            )
            n_display = len(tracks)
        else:
            annotated_frame = draw_detections(
                frame, boxes, classes, scores, class_names, line_width
            )
            n_display = len(boxes)
        
        # Add frame info
        mode_label = "Tracks" if tracker is not None else "Detections"
        info_text = f"Frame: {i + 1}/{len(frames)} | {mode_label}: {n_display}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        # Write frames to both videos
        writer_annotated.write(annotated_frame)
        writer_clean.write(frame)
        
        # Save text detections if requested
        if save_txt and txt_output_dir:
            txt_path = txt_output_dir / f"{frame_path.stem}.txt"
            with open(txt_path, "w") as f:
                if tracker is not None:
                    for t in tracks:
                        x1, y1, x2, y2 = t["ltrb"]
                        f.write(f"{t['track_id']} {t['class_id']} {x1} {y1} {x2} {y2}\n")
                else:
                    for box, cls, score in zip(boxes, classes, scores):
                        x1, y1, x2, y2 = box
                        f.write(f"{int(cls)} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {score:.4f}\n")
        
        # Show if requested
        if show:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrupted by user")
                break
        
        # Progress update
        if (i + 1) % 50 == 0 or (i + 1) == len(frames):
            print(f"    Progress: {i + 1}/{len(frames)} frames")
    
    writer_annotated.release()
    writer_clean.release()
    if show:
        cv2.destroyAllWindows()
    
    return {"frames": len(frames), "detections": total_detections, "annotated_path": annotated_path, "clean_path": clean_path}



def process_video_file(
    model: YOLO,
    video_path: Path,
    output_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    class_names: list[str],
    line_width: int,
    show: bool,
    save_txt: bool,
    device: str = "cpu",
    half: bool = False,
    max_frames: int | None = None,
    tracker=None,
) -> dict:
    """Process a video file and output annotated video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return {"frames": 0, "detections": 0}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"  Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create two video writers - one with annotations, one without
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Annotated video (with bounding boxes)
    annotated_path = output_path.parent / f"{output_path.stem}_annotated{output_path.suffix}"
    writer_annotated = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
    
    # Clean video (without bounding boxes)
    clean_path = output_path.parent / f"{output_path.stem}_clean{output_path.suffix}"
    writer_clean = cv2.VideoWriter(str(clean_path), fourcc, fps, (width, height))
    
    # Create text output directory if needed
    txt_output_dir = None
    if save_txt:
        txt_output_dir = output_path.parent / f"{output_path.stem}_labels"
        txt_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_detections = 0
    frame_count = 0
    
    # Reset tracker state for this video
    if tracker is not None:
        tracker.reset()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        # Run inference
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )[0]
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        total_detections += len(boxes)
        
        # Draw with tracker or plain detections
        if tracker is not None:
            tracks = tracker.update(frame, boxes, scores, classes)
            annotated_frame = tracker.draw_tracks(
                frame, tracks, class_names, line_width,
            )
            n_display = len(tracks)
        else:
            annotated_frame = draw_detections(
                frame, boxes, classes, scores, class_names, line_width
            )
            n_display = len(boxes)
        
        # Add frame info
        mode_label = "Tracks" if tracker is not None else "Detections"
        info_text = f"Frame: {frame_count + 1}/{total_frames} | {mode_label}: {n_display}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        # Write frames to both videos
        writer_annotated.write(annotated_frame)
        writer_clean.write(frame)
        
        # Save text detections if requested
        if save_txt and txt_output_dir:
            txt_path = txt_output_dir / f"frame_{frame_count:06d}.txt"
            with open(txt_path, "w") as f:
                if tracker is not None:
                    for t in tracks:
                        x1, y1, x2, y2 = t["ltrb"]
                        f.write(f"{t['track_id']} {t['class_id']} {x1} {y1} {x2} {y2}\n")
                else:
                    for box, cls, score in zip(boxes, classes, scores):
                        x1, y1, x2, y2 = box
                        f.write(f"{int(cls)} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {score:.4f}\n")
        
        # Show if requested
        if show:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrupted by user")
                break
        
        frame_count += 1
        
        # Progress update
        if frame_count % 50 == 0 or frame_count == total_frames:
            print(f"    Progress: {frame_count}/{total_frames} frames")
    
    cap.release()
    writer_annotated.release()
    writer_clean.release()
    if show:
        cv2.destroyAllWindows()
    
    return {"frames": frame_count, "detections": total_detections, "annotated_path": annotated_path, "clean_path": clean_path}


def main():
    args = parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    # Load model
    print(f"\n{'=' * 60}")
    print("YOLOv8 Video Inference for VisDrone")
    print(f"{'=' * 60}")
    print(f"Loading model: {args.weights}")
    
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Determine class names
    if args.use_visdrone_classes:
        class_names = VISDRONE_CLASSES
        print(f"Using VisDrone classes ({len(class_names)} classes)")
    else:
        # Use model's class names (COCO or custom)
        class_names = model.names
        if isinstance(class_names, dict):
            class_names = [class_names[i] for i in sorted(class_names.keys())]
        print(f"Using model classes ({len(class_names)} classes)")
    
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    
    # Initialise DeepSORT tracker if requested
    tracker = None
    if args.track:
        try:
            from visdrone_toolkit.tracker import DeepSORTTracker
        except ImportError:
            print("Error: deep-sort-realtime package not found.")
            print("Install with: pip install deep-sort-realtime")
            return 1
        tracker = DeepSORTTracker(
            max_age=args.track_max_age,
            n_init=args.track_n_init,
            max_cosine_distance=args.track_max_cosine_distance,
        )
        print(f"Tracking: DeepSORT (max_age={args.track_max_age}, "
              f"n_init={args.track_n_init})")
    
    print(f"{'=' * 60}\n")
    
    # Determine input type and process
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    if input_path.is_file() and input_path.suffix.lower() in video_extensions:
        # Single video file
        print(f"Processing video: {input_path.name}")
        output_path = output_dir / f"{input_path.stem}_yolov8.mp4"
        
        stats = process_video_file(
            model=model,
            video_path=input_path,
            output_path=output_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            class_names=class_names,
            line_width=args.line_width,
            show=args.show,
            save_txt=args.save_txt,
            device=args.device,
            half=args.half,
            max_frames=args.max_frames,
            tracker=tracker,

        )
        
        print(f"\n✓ Saved annotated video: {stats['annotated_path']}")
        print(f"✓ Saved clean video: {stats['clean_path']}")
        print(f"  Frames: {stats['frames']}, Detections: {stats['detections']}")
    
    elif input_path.is_dir():
        if args.batch_sequences:
            # Process all subdirectories as sequences
            sequences = sorted([d for d in input_path.iterdir() if d.is_dir()])
            
            if len(sequences) == 0:
                print(f"No sequence directories found in {input_path}")
                return 1
            
            print(f"Found {len(sequences)} sequences to process\n")
            
            total_frames = 0
            total_detections = 0
            
            for seq_dir in sequences:
                print(f"Processing: {seq_dir.name}")
                output_path = output_dir / f"{seq_dir.name}_yolov8.mp4"
                
                stats = process_frame_sequence(
                    model=model,
                    sequence_dir=seq_dir,
                    output_path=output_path,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    fps=args.fps,
                    class_names=class_names,
                    line_width=args.line_width,
                    show=args.show,
                    save_txt=args.save_txt,
                    device=args.device,
                    half=args.half,
                    max_frames=args.max_frames,
                    tracker=tracker,

                )
                
                total_frames += stats["frames"]
                total_detections += stats["detections"]
                print(f"  ✓ Saved annotated: {stats['annotated_path'].name}")
                print(f"  ✓ Saved clean: {stats['clean_path'].name}\n")
            
            print(f"{'=' * 60}")
            print(f"✓ Processed {len(sequences)} sequences")
            print(f"  Total frames: {total_frames}")
            print(f"  Total detections: {total_detections}")
        
        else:
            # Single sequence directory
            frames = get_frame_files(input_path)
            
            if len(frames) > 0:
                # Treat as frame sequence
                print(f"Processing frame sequence: {input_path.name}")
                output_path = output_dir / f"{input_path.name}_yolov8.mp4"
                
                stats = process_frame_sequence(
                    model=model,
                    sequence_dir=input_path,
                    output_path=output_path,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    fps=args.fps,
                    class_names=class_names,
                    line_width=args.line_width,
                    show=args.show,
                    save_txt=args.save_txt,
                    device=args.device,
                    half=args.half,
                    max_frames=args.max_frames,
                    tracker=tracker,
                )
                
                print(f"\n✓ Saved annotated video: {stats['annotated_path']}")
                print(f"✓ Saved clean video: {stats['clean_path']}")
                print(f"  Frames: {stats['frames']}, Detections: {stats['detections']}")
            else:
                print(f"No frames found in {input_path}")
                print("Use --batch-sequences to process subdirectories")
                return 1
    
    else:
        print(f"Error: Unsupported input type: {input_path}")
        return 1
    
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
