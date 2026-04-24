import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo
from art.attacks.evasion import SNAL

os.environ["YOLO_SETTINGS_SYNC"] = "False"
os.environ["YOLO_OFFLINE"] = "1"

yolo_model = YOLO("yolov8n.pt")

def ultralytics_collector(estimator, x):
    import torch
    
    if torch.is_tensor(x):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    processed_images = []
    for i in range(x_np.shape[0]):
        img = np.transpose(x_np[i], (1, 2, 0))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        processed_images.append(img)

    results = yolo_model.predict(processed_images, conf=0.2, verbose=False)
    
    batch_patches = []
    batch_positions = []
    
    for i, res in enumerate(results):
        img_patches = []
        img_np = processed_images[i] 
        
        if len(res.boxes) > 0:
            boxes_np = res.boxes.xyxy.cpu().numpy()
            formatted_boxes = []
            
            for box in boxes_np:
                x1, y1, x2, y2 = map(int, box)
                formatted_boxes.append([x1, y1, x2, y2])
                
                patch = img_np[y1:y2, x1:x2].copy()
                if patch.size > 0:
                    patch_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                    img_patches.append(patch_t.to(estimator.device))
            
            # Successful detection: return (N, 4) LongTensor
            pos_tensor = torch.tensor(formatted_boxes, dtype=torch.int64).to(estimator.device)
        else:
            # NO DETECTION: Return an empty tensor with shape (0, 4)
            # This allows .T to unpack 4 empty lists instead of crashing
            pos_tensor = torch.zeros((0, 4), dtype=torch.int64).to(estimator.device)
        
        batch_patches.append(img_patches)
        batch_positions.append(pos_tensor)
        
    return batch_patches, batch_positions

# Setup Estimator
art_detector = PyTorchYolo(
    model=yolo_model.model,
    device_type="cuda" if torch.cuda.is_available() else "cpu",
    input_shape=(3, 640, 640),
    clip_values=(0.0, 1.0),
    channels_first=True
)

def prepare_data(image_dir: Path, max_images: int = None):
    """Load images from directory."""
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    
    if max_images is not None:
        paths = paths[:max_images]
    
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((640, 640))
        img_np = np.transpose(np.array(img).astype(np.float32) / 255.0, (2, 0, 1))
        images.append(img_np)
    
    print(f"Loaded {len(images)} images from {image_dir}")
    if len(images) > 0:
        print(f"First image stats - shape: {images[0].shape}, min: {images[0].min():.4f}, max: {images[0].max():.4f}")
            
    return np.stack(images), paths

def create_universal_candidates(num_candidates: int = 5):
    """Create a single set of universal patch candidates."""
    candidates = [torch.rand((3, 128, 128)).to(art_detector.device) for _ in range(num_candidates)]
    print(f"Created {num_candidates} universal patch candidates")
    return candidates

if __name__ == "__main__":
    input_dir = Path("data/VisDrone2019-DET-val/images")
    output_dir = Path("outputs/outputs_snal_universal")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load representative subset for universal patch optimization
    print("="*60)
    print("STEP 1: Loading representative subset for patch optimization")
    print("="*60)
    x_train_subset, train_paths = prepare_data(input_dir, max_images=20)  # Use first 20 images to optimize
    
    # Step 2: Create universal patch candidates (single set, shared across all images)
    print("\n" + "="*60)
    print("STEP 2: Creating universal patch candidates")
    print("="*60)
    universal_candidates = create_universal_candidates(num_candidates=5)
    
    # Step 3: Optimize universal patch on representative subset
    print("\n" + "="*60)
    print("STEP 3: Optimizing universal patch on representative subset")
    print("="*60)
    attack = SNAL(
        estimator=art_detector,
        candidates=universal_candidates,
        collector=ultralytics_collector,
        eps=0.4,
        max_iter=100,
        num_grid=16 
    )

    try:
        print(f"Training set shape: {x_train_subset.shape}")
        print(f"Training set min: {x_train_subset.min():.4f}, max: {x_train_subset.max():.4f}")
        
        # Optimize patch on subset
        x_train_adv = attack.generate(x=x_train_subset)
        
        print(f"Adversarial subset shape: {x_train_adv.shape}")
        print(f"Adversarial subset min: {x_train_adv.min():.4f}, max: {x_train_adv.max():.4f}")
        print(f"Difference min: {(x_train_adv - x_train_subset).min():.6f}, max: {(x_train_adv - x_train_subset).max():.6f}")
        
        # Extract the universal patch (average difference across training subset)
        universal_patch = np.mean(x_train_adv - x_train_subset, axis=0, keepdims=True)
        print(f"Universal patch shape: {universal_patch.shape}")
        print(f"Universal patch min: {universal_patch.min():.6f}, max: {universal_patch.max():.6f}")
        print(f"Universal patch mean: {universal_patch.mean():.6f}")
        
        # Find the patch with maximum perturbation strength (more aggressive)
        max_perturb = np.max(np.abs(x_train_adv - x_train_subset), axis=0, keepdims=True)
        print(f"Max perturbation min: {max_perturb.min():.6f}, max: {max_perturb.max():.6f}")
        print(f"Max perturbation mean: {max_perturb.mean():.6f}")
        
        # Use a blend: mean direction with max magnitude approach
        # This preserves adversarial direction while applying stronger perturbation
        patch_direction = np.sign(universal_patch)
        universal_patch_strong = patch_direction * max_perturb * 0.8  # Use 80% of max perturbation strength
        print(f"Strong universal patch min: {universal_patch_strong.min():.6f}, max: {universal_patch_strong.max():.6f}")
        print(f"Strong universal patch mean: {universal_patch_strong.mean():.6f}")
        
        # Save the universal patch as PNG
        # Convert from CHW to HWC format and normalize to [0, 255]
        patch_hwc = np.transpose(universal_patch_strong[0], (1, 2, 0))
        patch_uint8 = np.clip((patch_hwc + 0.5) * 255, 0, 255).astype(np.uint8)  # Shift to center at 128 for visibility
        patch_img = Image.fromarray(patch_uint8)
        patch_path = output_dir / "steal_now_attack_later_universal_patch.png"
        patch_img.save(patch_path)
        print(f"Universal patch saved to {patch_path}")
        
        # Step 4: Load all test images
        print("\n" + "="*60)
        print("STEP 4: Loading all test images")
        print("="*60)
        x_test, test_paths = prepare_data(input_dir, max_images=None)
        
        # Step 5: Apply universal patch to all test images
        print("\n" + "="*60)
        print("STEP 5: Applying universal patch to all test images")
        print("="*60)
        x_test_adv = np.clip(x_test + universal_patch_strong, 0, 1)
        
        print(f"Test set with patch shape: {x_test_adv.shape}")
        print(f"Test set with patch min: {x_test_adv.min():.4f}, max: {x_test_adv.max():.4f}")
        
        # Step 6: Evaluate effectiveness on all test images
        print("\n" + "="*60)
        print("STEP 6: Evaluating patch effectiveness on all test images")
        print("="*60)
        results = []
        
        for i in range(len(x_test_adv)):
            if i % 50 == 0:
                print(f"Processing image {i}/{len(x_test_adv)}...")
                
            adv_img_hwc = np.transpose(x_test_adv[i], (1, 2, 0))
            clean_img_hwc = np.transpose(x_test[i], (1, 2, 0))

            adv_img_uint8 = (np.clip(adv_img_hwc, 0, 1) * 255).astype(np.uint8)
            clean_img_uint8 = (np.clip(clean_img_hwc, 0, 1) * 255).astype(np.uint8)
            
            # Save patched image
            Image.fromarray(adv_img_uint8).save(output_dir / f"patched_{test_paths[i].name}")
            
            # Evaluate detections
            clean_res = yolo_model.predict(clean_img_uint8, conf=0.2, verbose=False)[0]
            adv_res = yolo_model.predict(adv_img_uint8, conf=0.2, verbose=False)[0]
            
            clean_count = len(clean_res.boxes)
            adv_count = len(adv_res.boxes)
            
            results.append({
                'image_name': test_paths[i].name,
                'clean_detections': clean_count,
                'patched_detections': adv_count,
                'detection_reduction': clean_count - adv_count
            })
        
        # Summary statistics
        print("\n" + "="*60)
        print("STEP 7: Results Summary")
        print("="*60)
        df = pd.DataFrame(results)
        csv_path = output_dir / "universal_patch_results.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\nTotal test images: {len(results)}")
        print(f"Average clean detections: {df['clean_detections'].mean():.2f}")
        print(f"Average patched detections: {df['patched_detections'].mean():.2f}")
        print(f"Average detection reduction: {df['detection_reduction'].mean():.2f}")
        print(f"Success rate (detection reduced): {(df['detection_reduction'] > 0).sum() / len(df) * 100:.1f}%")
        print(f"\nResults exported to {csv_path}")
        
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback
        traceback.print_exc()