"""
Command-line tool for converting VisDrone annotations to other formats.

Supports conversion to:
- COCO format (JSON)
- YOLO format (TXT)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visdrone_toolkit.converters import convert_to_coco, convert_to_yolo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone annotations to other formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to COCO format
  python convert_annotations.py --format coco \\
      --image-dir data/images \\
      --annotation-dir data/annotations \\
      --output annotations.json

  # Convert to YOLO format
  python convert_annotations.py --format yolo \\
      --image-dir data/images \\
      --annotation-dir data/annotations \\
      --output-dir data/yolo_annotations

""",
    )

    # Required arguments
    parser.add_argument("--format", required=True, choices=["coco", "yolo"], help="Output format")
    parser.add_argument("--image-dir", required=True, help="Path to images directory")
    parser.add_argument(
        "--annotation-dir", required=True, help="Path to VisDrone annotations directory"
    )

    # Output arguments
    parser.add_argument("--output", help="Output file path (for COCO)")
    parser.add_argument("--output-dir", help="Output directory (for YOLO)")

    # Filtering options
    parser.add_argument(
        "--keep-ignored", action="store_true", help="Keep boxes with score=0 (default: filter out)"
    )
    parser.add_argument(
        "--keep-crowd", action="store_true", help="Keep crowd/ignored regions (default: filter out)"
    )

    # YOLO-specific options
    parser.add_argument("--yolo-images-dir", help="YOLO: Directory to copy/link images")
    parser.add_argument("--no-yaml", action="store_true", help="YOLO: Don't create dataset.yaml")

    # Validation
    parser.add_argument("--validate", action="store_true", help="Validate output after conversion")

    return parser.parse_args()


def main():
    args = parse_args()

    # Verify input directories exist
    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)

    if not image_dir.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return 1

    if not annotation_dir.exists():
        print(f"Error: Annotation directory does not exist: {annotation_dir}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"Converting VisDrone Annotations to {args.format.upper()} Format")
    print(f"{'=' * 60}")
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Filter ignored boxes: {not args.keep_ignored}")
    print(f"Filter crowd regions: {not args.keep_crowd}")
    print(f"{'=' * 60}\n")

    try:
        if args.format == "coco":
            # COCO format conversion
            if not args.output:
                print("Error: --output is required for COCO format")
                return 1

            convert_to_coco(
                image_dir=str(image_dir),
                annotation_dir=str(annotation_dir),
                output_json=args.output,
                filter_ignored=not args.keep_ignored,
                filter_crowd=not args.keep_crowd,
            )

            # Validate if requested
            if args.validate:
                print("\nValidating COCO format...")
                from visdrone_toolkit.converters.visdrone_to_coco import validate_coco_format

                validate_coco_format(args.output)

        elif args.format == "yolo":
            # YOLO format conversion
            if not args.output_dir:
                print("Error: --output-dir is required for YOLO format")
                return 1

            convert_to_yolo(
                image_dir=str(image_dir),
                annotation_dir=str(annotation_dir),
                output_dir=args.output_dir,
                output_images_dir=args.yolo_images_dir,
                filter_ignored=not args.keep_ignored,
                filter_crowd=not args.keep_crowd,
                create_yaml=not args.no_yaml,
            )

            # Validate if requested
            if args.validate:
                print("\nValidating YOLO format...")
                from visdrone_toolkit.converters.visdrone_to_yolo import validate_yolo_format

                validate_yolo_format(args.output_dir)

        print(f"\n{'=' * 60}")
        print("✓ Conversion completed successfully!")
        print(f"{'=' * 60}\n")

        return 0

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"✗ Error during conversion: {e}")
        print(f"{'=' * 60}\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())