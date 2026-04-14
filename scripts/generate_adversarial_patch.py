import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo
from art.attacks.evasion import RobustDPatch

os.environ["YOLO_SETTINGS_SYNC"] = "False"
os.environ["YOLO_OFFLINE"] = "1"

yolo = YOLO("yolov8n.pt")

class UltralyticsYoloEstimator(PyTorchYolo):
    def loss_gradient(self, x: np.ndarray, y, **kwargs) -> np.ndarray:
        device = next(self.model.parameters()).device

        # Bypass PyTorchYoloLossWrapper: in eval mode it returns a non-differentiable
        # list[dict] after postprocessing. self.model.model is the underlying
        # ultralytics DetectionModel whose eval output retains gradients.
        raw_model = self.model.model if hasattr(self.model, "model") else self.model
        raw_model.eval()

        with torch.set_grad_enabled(True):
            x_tensor = torch.from_numpy(x).to(device).requires_grad_(True)

            # DetectionModel eval returns (y_tensor, preds_dict) where
            # preds_dict = {"boxes": ..., "scores": ..., "feats": ...}
            # preds_dict["scores"] shape: (batch, nc, na) — raw class logits
            out = raw_model(x_tensor)

            if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict) and "scores" in out[1]:
                # Preferred path: raw class logits, fully differentiable
                loss = out[1]["scores"].sigmoid().mean()
            elif isinstance(out, (list, tuple)):
                # Fallback: processed inference tensor (batch, 4+nc, na)
                # YOLOv8 has no objectness score — class scores start at index 4
                loss = out[0][:, 4:, :].mean()
            elif isinstance(out, torch.Tensor):
                loss = out[:, 4:, :].mean()
            else:
                raise RuntimeError(f"Unexpected model output type: {type(out)}")

            raw_model.zero_grad()
            loss.backward()

            if x_tensor.grad is None:
                raise RuntimeError("Gradient is None — gradient chain may be broken")

            return x_tensor.grad.detach().cpu().numpy()


art_detector = UltralyticsYoloEstimator(
    model=yolo.model,
    model_name="yolov8n",
    input_shape=(3, 640, 640),
    clip_values=(0.0, 1.0),
    channels_first=True,
    is_ultralytics=True,
)

def load_visdrone_val_batch(val_images_dir: Path, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    image_paths = sorted(
        [p for p in val_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Expected at least 1 image in {val_images_dir}, found {len(image_paths)}"
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
        return img.size[::-1]  # transpose


if __name__ == "__main__":
    val_images_dir = Path("data/VisDrone2019-DET-val/images")
    val_annotations_dir = Path("data/VisDrone2019-DET-val/annotations")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    images_np = load_visdrone_val_batch(val_images_dir)
    print(f"Loaded batch shape: {images_np.shape}")

    # Get image file names for annotation loading
    image_paths = sorted(
        [p for p in val_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    image_names = [p.name for p in image_paths]
    print(f"Processing {len(image_paths)} images")

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
        print(f"Passed. Got {len(preds)} predictions")
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

    # Generate adversarial patch with RobustDPatch
    print("\n=== Generating adversarial patch (RobustDPatch) ===")
    PATCH_SHAPE = (3, 150, 150)   # channels-first
    PATCH_LOCATION = (50, 250)    # (y_start, x_start) — object-dense region

    attack = RobustDPatch(
        estimator=art_detector,
        patch_shape=PATCH_SHAPE,
        learning_rate=0.01,
        max_iter=50,
        batch_size=4,
        verbose=True,
    )
    patch = attack.generate(x=images_np)
    print(f"Generated patch. Shape: {patch.shape}, Min: {patch.min():.4f}, Max: {patch.max():.4f}")

    # Save patch as PNG
    patch_uint8 = (np.transpose(patch, (1, 2, 0)) * 255).astype(np.uint8)
    Image.fromarray(patch_uint8).save(output_dir / "adversarial_patch.png")
    print(f"Saved patch to {output_dir / 'adversarial_patch.png'}")

    # Apply patch to all images and save results
    print("\nApplying patch to all images...")
    patched_images = np.array([
        apply_patch_to_image(images_np[i], patch, location=PATCH_LOCATION)
        for i in range(len(images_np))
    ])
    
    # Save all patched images
    patched_output_dir = output_dir / "patched_images"
    patched_output_dir.mkdir(exist_ok=True)
    for i in range(len(image_paths)):
        patched_uint8 = (np.transpose(patched_images[i], (1, 2, 0)) * 255).astype(np.uint8)
        image_name = image_paths[i].stem
        Image.fromarray(patched_uint8).save(patched_output_dir / f"{image_name}_patched.png")
    print(f"Saved {len(image_paths)} patched images to {patched_output_dir}/")

    # Before/after detection comparison
    print("\n    Before/After Detection Comparison    ")
    clean_preds = art_detector.predict(images_np)
    clean_counts = [len(pred.get("boxes", np.array([]))) for pred in clean_preds]
    total_clean = sum(clean_counts)
    print(f"Total detections (clean): {total_clean} across {len(images_np)} images")

    patched_preds = art_detector.predict(patched_images)
    patched_counts = [len(pred.get("boxes", np.array([]))) for pred in patched_preds]
    total_patched = sum(patched_counts)
    print(f"Total detections (patched): {total_patched} across {len(patched_images)} images")

    delta = total_clean - total_patched
    pct = 100.0 * delta / max(total_clean, 1)
    print(f"Delta: {delta} detections ({pct:.1f}%)")
    
    # Per-image comparison
    print("\n=== Per-Image Comparison ===")
    print(f"{'Image':<40} {'Clean':>8} {'Patched':>8} {'Change':>8}")
    print("-" * 65)
    for i, image_path in enumerate(image_paths):
        clean_det = clean_counts[i]
        patched_det = patched_counts[i]
        change = clean_det - patched_det
        print(f"{image_path.name:<40} {clean_det:>8} {patched_det:>8} {change:>8}")
