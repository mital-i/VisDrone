"""Convert VisDrone annotations to COCO format.

COCO format is widely used and compatible with many modern detection frameworks
including torchvision, Detectron2, and MMDetection.

"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image
from tqdm import tqdm


def convert_to_coco(
    image_dir: Union[str, Path],
    annotation_dir: Union[str, Path],
    output_json: Union[str, Path],
    filter_ignored: bool = True,
    filter_crowd: bool = True,
    description: str = "VisDrone Dataset in COCO format",
) -> Dict:
    """Convert VisDrone annotations to COCO format.

    Args:
        image_dir: Path to images directory
        annotation_dir: Path to VisDrone annotations directory
        output_json: Path to output COCO JSON file
        filter_ignored: Filter boxes with score=0
        filter_crowd: Filter crowd/ignored regions (category=0)
        description: Dataset description

    Returns:
        COCO format dictionary

    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    # Validate directories exist
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not annotation_dir.exists():
        raise ValueError(f"Annotation directory does not exist: {annotation_dir}")

    # VisDrone categories
    categories = [
        {"id": 0, "name": "ignored-regions", "supercategory": "none"},
        {"id": 1, "name": "pedestrian", "supercategory": "person"},
        {"id": 2, "name": "people", "supercategory": "person"},
        {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 4, "name": "car", "supercategory": "vehicle"},
        {"id": 5, "name": "van", "supercategory": "vehicle"},
        {"id": 6, "name": "truck", "supercategory": "vehicle"},
        {"id": 7, "name": "tricycle", "supercategory": "vehicle"},
        {"id": 8, "name": "awning-tricycle", "supercategory": "vehicle"},
        {"id": 9, "name": "bus", "supercategory": "vehicle"},
        {"id": 10, "name": "motor", "supercategory": "vehicle"},
        {"id": 11, "name": "others", "supercategory": "none"},
    ]

    # Filter categories if needed
    if filter_crowd:
        categories = [cat for cat in categories if cat["id"] != 0]

    # Initialize COCO structure
    coco: Dict[str, Any] = {
        "info": {
            "description": description,
            "url": "https://github.com/VisDrone/VisDrone-Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "VisDrone",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "",
            }
        ],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    # Get all image files
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Converting {len(image_files)} images to COCO format...")

    annotation_id = 1

    for image_id, image_path in enumerate(tqdm(image_files), start=1):
        # Load image to get dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"Warning: Could not open {image_path}: {e}")
            continue

        # Add image info
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": "",
            }
        )

        # Parse annotations
        ann_path = annotation_dir / (image_path.stem + ".txt")
        if not ann_path.exists():
            continue

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
                truncation = int(parts[6])
                occlusion = int(parts[7])

                # Filter ignored boxes
                if filter_ignored and score == 0:
                    continue

                # Filter crowd
                if filter_crowd and category == 0:
                    continue

                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                # COCO format: [x, y, width, height]
                bbox = [bbox_left, bbox_top, bbox_width, bbox_height]
                area = bbox_width * bbox_height

                # Add annotation
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 1 if category == 0 else 0,
                        "segmentation": [],  # VisDrone doesn't have segmentation
                        "truncation": truncation,
                        "occlusion": occlusion,
                    }
                )
                annotation_id += 1

    # Save to JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nCOCO format saved to {output_path}")
    print(f"Total images: {len(coco['images'])}")
    print(f"Total annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")

    return coco


def validate_coco_format(coco_json: str) -> bool:
    """Validate COCO format JSON file.

    Args:
        coco_json: Path to COCO JSON file

    Returns:
        True if valid, False otherwise

    """
    try:
        with open(coco_json) as f:
            coco = json.load(f)

        required_keys = ["info", "licenses", "categories", "images", "annotations"]
        for key in required_keys:
            if key not in coco:
                print(f"Missing required key: {key}")
                return False

        # Check categories
        if len(coco["categories"]) == 0:
            print("No categories found")
            return False

        # Check images
        if len(coco["images"]) == 0:
            print("No images found")
            return False

        # Check annotations
        image_ids = {img["id"] for img in coco["images"]}
        category_ids = {cat["id"] for cat in coco["categories"]}

        for ann in coco["annotations"]:
            if ann["image_id"] not in image_ids:
                print(f"Invalid image_id: {ann['image_id']}")
                return False
            if ann["category_id"] not in category_ids:
                print(f"Invalid category_id: {ann['category_id']}")
                return False

        print("COCO format is valid!")
        print(f"Images: {len(coco['images'])}")
        print(f"Annotations: {len(coco['annotations'])}")
        print(f"Categories: {len(coco['categories'])}")

        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert VisDrone to COCO format")
    parser.add_argument("--image_dir", required=True, help="Path to images directory")
    parser.add_argument("--annotation_dir", required=True, help="Path to annotations directory")
    parser.add_argument("--output", required=True, help="Output COCO JSON file")
    parser.add_argument("--no-filter-ignored", action="store_true", help="Keep ignored boxes")
    parser.add_argument("--no-filter-crowd", action="store_true", help="Keep crowd regions")
    parser.add_argument("--validate", action="store_true", help="Validate output")

    args = parser.parse_args()

    convert_to_coco(
        args.image_dir,
        args.annotation_dir,
        args.output,
        filter_ignored=not args.no_filter_ignored,
        filter_crowd=not args.no_filter_crowd,
    )

    if args.validate:
        validate_coco_format(args.output)
