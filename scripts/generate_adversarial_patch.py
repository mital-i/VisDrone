import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo
from art.attacks.evasion import RobustDPatch

os.environ["YOLO_SETTINGS_SYNC"] = "False"
os.environ["YOLO_OFFLINE"] = "1"

yolo = YOLO("yolov8n.pt")
yolo.model.train()  # Ensure model is in training mode for gradient computation

art_detector = PyTorchYolo(
    model=yolo.model,
    model_name="yolov8n",
    input_shape=(3, 640, 640),
    clip_values=(0.0, 1.0),
    channels_first=True,
    is_ultralytics=True,
)

def load_visdrone_val_batch(val_images_dir: Path, n: int = 4, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    image_paths = sorted(
        [p for p in val_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )[:n]
    if len(image_paths) < n:
        raise FileNotFoundError(
            f"Expected at least {n} images in {val_images_dir}, found {len(image_paths)}"
        )

    batch = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(size, Image.BILINEAR)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))
        batch.append(image_arr)

    return np.stack(batch, axis=0)


def load_visdrone_annotations_art_format(
    annotations_dir: Path, image_names: list[str], orig_size: tuple[int, int], target_size: tuple[int, int] = (640, 640)
) -> list[dict]:
    """
    Load VisDrone annotations and convert to ART format.
    
    VisDrone format: bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
    ART format: {"boxes": [[x1, y1, x2, y2], ...], "labels": [cls_id, ...], "scores": [1.0, ...]}
    
    Skips category 0 (ignored) and 11 (others).
    Scales boxes to target_size.
    """
    labels_list = []
    target_h, target_w = target_size
    orig_h, orig_w = orig_size
    scale_x, scale_y = target_w / orig_w, target_h / orig_h
    
    for image_name in image_names:
        ann_file = annotations_dir / image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        
        boxes = []
        labels = []
        scores = []
        
        if ann_file.exists():
            with open(ann_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    
                    bbox_left, bbox_top, bbox_width, bbox_height = map(float, parts[:4])
                    score = float(parts[4])
                    category = int(float(parts[5]))
                    
                    # Skip category 0 (ignored) and 11 (others)
                    if category == 0 or category == 11:
                        continue
                    
                    # Scale to target image size
                    x1 = bbox_left * scale_x
                    y1 = bbox_top * scale_y
                    x2 = (bbox_left + bbox_width) * scale_x
                    y2 = (bbox_top + bbox_height) * scale_y
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(category)
                    scores.append(1.0)
        
        labels_list.append({
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64),
            "scores": np.array(scores, dtype=np.float32) if scores else np.zeros(0, dtype=np.float32),
        })
    
    return labels_list


def apply_patch_to_image(image_np: np.ndarray, patch: np.ndarray, location: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Apply adversarial patch to an image (channels-first format, patch also channels-first)."""
    patched = image_np.copy()
    patch_c, patch_h, patch_w = patch.shape
    y_start, x_start = location
    patched[:, y_start:y_start + patch_h, x_start:x_start + patch_w] = patch
    return patched


def get_image_original_size(image_path: Path) -> tuple[int, int]:
    """Get original image dimensions before resizing."""
    with Image.open(image_path) as img:
        return img.size[::-1]  # (width, height) -> (height, width)


if __name__ == "__main__":
    val_images_dir = Path("data/VisDrone2019-DET-val/images")
    val_annotations_dir = Path("data/VisDrone2019-DET-val/annotations")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Load batch of 4 images
    images_np = load_visdrone_val_batch(val_images_dir, n=4)
    print("Loaded batch shape:", images_np.shape)
    
    # Get image file names for annotation loading
    image_paths = sorted(
        [p for p in val_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )[:4]
    image_names = [p.name for p in image_paths]
    
    # Get original size from first image
    orig_size = get_image_original_size(image_paths[0])
    
    # Load annotations in ART format
    labels_list = load_visdrone_annotations_art_format(
        val_annotations_dir, image_names, orig_size=orig_size, target_size=(640, 640)
    )
    print(f"Loaded annotations for {len(labels_list)} images")
    
    # Test basic prediction
    try:
        print("\nTesting basic prediction...")
        preds = art_detector.predict(images_np)
        print(f"✓ Prediction works. Got {len(preds)} predictions")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        sys.exit(1)
    
    # Try gradient computation
    try:
        print("\nTesting gradient computation...")
        grads = art_detector.loss_gradient(x=images_np[:1], y=[labels_list[0]])
        print(f"✓ Gradient computation works. Shape: {grads.shape}")
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        print("\nNote: ART RobustDPatch may not be fully compatible with this ultralytics/PyTorch version.")
        print("Falling back to simple patch generation demo...")
        
        # Simple fallback: create a random patch for demonstration
        patch = np.random.uniform(0.3, 0.7, size=(3, 40, 40)).astype(np.float32)
        print(f"Generated demo patch. Shape: {patch.shape}, Min: {patch.min():.4f}, Max: {patch.max():.4f}")
        
        # Save patch as PNG
        patch_uint8 = (patch * 255).astype(np.uint8)
        patch_uint8 = np.transpose(patch_uint8, (1, 2, 0))
        patch_img = Image.fromarray(patch_uint8)
        patch_img.save(output_dir / "adversarial_patch.png")
        print(f"Saved demo patch to outputs/adversarial_patch.png")
        
        # Save patched sample images
        for i in range(min(2, len(image_paths))):
            patched_img_np = apply_patch_to_image(images_np[i], patch, location=(0, 0))
            patched_img_uint8 = (np.transpose(patched_img_np, (1, 2, 0)) * 255).astype(np.uint8)
            patched_img_pil = Image.fromarray(patched_img_uint8)
            patched_img_pil.save(output_dir / f"patched_sample_{i}.png")
        print(f"Saved patched samples to outputs/patched_sample_*.png")
        
        # Before/After detection comparison
        print("\n=== Before/After Detection Comparison (Demo Patch) ===")
        
        # Predict on clean images
        clean_preds = art_detector.predict(images_np)
        clean_counts = [len(pred.get("boxes", np.array([]))) for pred in clean_preds]
        total_clean = sum(clean_counts)
        print(f"Total detections (clean): {total_clean} across {len(images_np)} images")
        
        # Apply patch to all images and predict
        patched_images = np.array([
            apply_patch_to_image(images_np[i], patch, location=(0, 0))
            for i in range(len(images_np))
        ])
        
        patched_preds = art_detector.predict(patched_images)
        patched_counts = [len(pred.get("boxes", np.array([]))) for pred in patched_preds]
        total_patched = sum(patched_counts)
        print(f"Total detections (patched): {total_patched} across {len(patched_images)} images")
        delta = total_clean - total_patched
        pct = 100 * delta / max(total_clean, 1) if total_clean > 0 else 0
        print(f"Delta: {delta} detections ({pct:.1f}%)")

