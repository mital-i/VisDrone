"""Convert VisDrone annotations to YOLO format.

YOLO format is popular for YOLOv5, YOLOv8, and other YOLO-based detectors.
Each annotation file contains normalized coordinates: class x_center y_center width height

"""

from pathlib import Path
from typing import Optional, Union

import yaml
from PIL import Image
from tqdm import tqdm


def convert_to_yolo(
    image_dir: Union[str, Path],
    annotation_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_images_dir: Optional[Union[str, Path]] = None,
    filter_ignored: bool = True,
    filter_crowd: bool = True,
    create_yaml: bool = True,
) -> None:
    """Convert VisDrone annotations to YOLO format.

    Args:
        image_dir: Path to images directory
        annotation_dir: Path to VisDrone annotations directory
        output_dir: Path to output directory for YOLO annotations
        output_images_dir: Optional path to copy/symlink images (if None, uses image_dir)
        filter_ignored: Filter boxes with score=0
        filter_crowd: Filter crowd/ignored regions (category=0)
        create_yaml: Create dataset.yaml for YOLO training

    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    # Validate directories exist
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not annotation_dir.exists():
        raise ValueError(f"Annotation directory does not exist: {annotation_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_images_dir:
        output_images_dir = Path(output_images_dir)
        output_images_dir.mkdir(parents=True, exist_ok=True)

    # VisDrone class names for YOLO (10 classes: categories 1-10)
    # Excludes: 0 (ignored-regions) and 11 (others)
    yolo_class_names = [
        "pedestrian",       # VisDrone 1 → YOLO 0
        "people",           # VisDrone 2 → YOLO 1
        "bicycle",          # VisDrone 3 → YOLO 2
        "car",              # VisDrone 4 → YOLO 3
        "van",              # VisDrone 5 → YOLO 4
        "truck",            # VisDrone 6 → YOLO 5
        "tricycle",         # VisDrone 7 → YOLO 6
        "awning-tricycle",  # VisDrone 8 → YOLO 7
        "bus",              # VisDrone 9 → YOLO 8
        "motor",            # VisDrone 10 → YOLO 9
    ]

    # Create class mapping (VisDrone category → YOLO class_id)
    # Maps categories 1-10 to YOLO 0-9, excludes 0 (ignored) and 11 (others)
    class_mapping = {i: i - 1 for i in range(1, 11)}

    # Get all image files
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Converting {len(image_files)} images to YOLO format...")

    converted_count = 0

    for image_path in tqdm(image_files):
        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Could not open {image_path}: {e}")
            continue

        # Parse VisDrone annotations
        ann_path = annotation_dir / (image_path.stem + ".txt")
        if not ann_path.exists():
            # Create empty annotation file
            output_ann_path = output_dir / (image_path.stem + ".txt")
            output_ann_path.write_text("")
            converted_count += 1
            continue

        yolo_annotations = []

        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 8:
                    continue

                bbox_left = int(parts[0])
                bbox_top = int(parts[1])
                bbox_width = int(parts[2])
                bbox_height = int(parts[3])
                score = int(parts[4])
                category = int(parts[5])

                # Filter ignored boxes
                if filter_ignored and score == 0:
                    continue

                # Filter ignored-regions (0) and others (11) - only keep categories 1-10
                if filter_crowd and (category == 0 or category == 11):
                    continue

                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                # Convert to YOLO format (normalized)
                x_center = (bbox_left + bbox_width / 2) / img_width
                y_center = (bbox_top + bbox_height / 2) / img_height
                norm_width = bbox_width / img_width
                norm_height = bbox_height / img_height

                # Ensure values are in [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))

                # Get YOLO class id
                yolo_class_id = class_mapping.get(category)
                if yolo_class_id is None:
                    continue

                # YOLO format: class x_center y_center width height
                yolo_annotations.append(
                    f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                )

        # Write YOLO annotation file
        output_ann_path = output_dir / (image_path.stem + ".txt")
        output_ann_path.write_text("\n".join(yolo_annotations))
        converted_count += 1

        # Copy or symlink images if output_images_dir is specified
        if output_images_dir:
            output_img_path = output_images_dir / image_path.name
            if not output_img_path.exists():
                try:
                    output_img_path.symlink_to(image_path)
                except OSError:
                    # If symlink fails, copy the file
                    import shutil

                    shutil.copy2(image_path, output_img_path)

    print(f"\nConverted {converted_count} images to YOLO format")
    print(f"Annotations saved to {output_dir}")

    if output_images_dir:
        print(f"Images linked/copied to {output_images_dir}")

    # Create dataset.yaml for YOLO training
    if create_yaml:
        yaml_path: Path = output_dir.parent / "dataset.yaml"

        yaml_content = {
            "path": str(output_dir.parent.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(yolo_class_names),
            "names": yolo_class_names,
            "download": "https://github.com/VisDrone/VisDrone-Dataset",
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        print(f"Dataset YAML saved to {yaml_path}")


def validate_yolo_format(annotation_dir: Union[str, Path]) -> bool:
    """Validate YOLO format annotation files.

    Args:
        annotation_dir: Path to YOLO annotations directory

    Returns:
        True if valid, False otherwise

    """
    annotation_dir = Path(annotation_dir)
    ann_files = list(annotation_dir.glob("*.txt"))

    if len(ann_files) == 0:
        print(f"No annotation files found in {annotation_dir}")
        return False

    print(f"Validating {len(ann_files)} YOLO annotation files...")

    valid_count = 0
    error_count = 0

    for ann_file in ann_files:
        try:
            with open(ann_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        print(
                            f"Error in {ann_file.name} line {line_num}: Expected 5 values, got {len(parts)}"
                        )
                        error_count += 1
                        continue

                    _ = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Check normalized coordinates are in [0, 1]
                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 <= width <= 1
                        and 0 <= height <= 1
                    ):
                        print(
                            f"Error in {ann_file.name} line {line_num}: Coordinates out of bounds"
                        )
                        error_count += 1
                        continue

            valid_count += 1

        except Exception as e:
            print(f"Error validating {ann_file.name}: {e}")
            error_count += 1

    print("\nValidation complete:")
    print(f"Valid files: {valid_count}")
    print(f"Errors: {error_count}")

    return error_count == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert VisDrone to YOLO format")
    parser.add_argument("--image_dir", required=True, help="Path to images directory")
    parser.add_argument("--annotation_dir", required=True, help="Path to annotations directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for YOLO annotations")
    parser.add_argument("--output_images_dir", help="Output directory for images (optional)")
    parser.add_argument("--no-filter-ignored", action="store_true", help="Keep ignored boxes")
    parser.add_argument("--no-filter-crowd", action="store_true", help="Keep crowd regions")
    parser.add_argument("--no-yaml", action="store_true", help="Don't create dataset.yaml")
    parser.add_argument("--validate", action="store_true", help="Validate output")

    args = parser.parse_args()

    convert_to_yolo(
        args.image_dir,
        args.annotation_dir,
        args.output_dir,
        output_images_dir=args.output_images_dir,
        filter_ignored=not args.no_filter_ignored,
        filter_crowd=not args.no_filter_crowd,
        create_yaml=not args.no_yaml,
    )

    if args.validate:
        validate_yolo_format(args.output_dir)
