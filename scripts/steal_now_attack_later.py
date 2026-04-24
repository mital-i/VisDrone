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

def prepare_data(image_dir: Path):
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((640, 640))
        img_np = np.transpose(np.array(img).astype(np.float32) / 255.0, (2, 0, 1))
        images.append(img_np)
    
    candidates = [torch.rand((3, 128, 128)).to(art_detector.device) for _ in range(len(paths))]
    
    print(f"Loaded {len(images)} images from {image_dir}")
    if len(images) > 0:
        print(f"First image stats - shape: {images[0].shape}, min: {images[0].min():.4f}, max: {images[0].max():.4f}")
            
    return np.stack(images), candidates, paths

if __name__ == "__main__":
    input_dir = Path("data/VisDrone2019-DET-val/images")
    output_dir = Path("outputs/outputs_snal")
    output_dir.mkdir(exist_ok=True)

    x_test, candidates, image_paths = prepare_data(input_dir)

    attack = SNAL(
        estimator=art_detector,
        candidates=candidates,
        collector=ultralytics_collector,
        eps=0.1,
        max_iter=50,
        num_grid=16 
    )

    print("\n\n Running SNAL Attack: ")
    try:
        print(f"Initial x_test shape: {x_test.shape}")
        print(f"x_test min: {x_test.min():.4f}, max: {x_test.max():.4f}")
        
        x_adv = attack.generate(x=x_test)
        
        print(f"x_adv shape: {x_adv.shape}")
        print(f"x_adv min: {x_adv.min():.4f}, max: {x_adv.max():.4f}")
        print(f"Difference (x_adv - x_test) min: {(x_adv - x_test).min():.6f}, max: {(x_adv - x_test).max():.6f}")
        print(f"Mean difference (x_adv - x_test): {np.mean(x_adv - x_test):.6f}")
        
        results = []
                
        for i in range(len(x_adv)):
            print(f"\nProcessing image {i}...")
            adv_img_hwc = np.transpose(x_adv[i], (1, 2, 0))
            clean_img_hwc = np.transpose(x_test[i], (1, 2, 0))

            adv_img_uint8 = (np.clip(adv_img_hwc, 0, 1) * 255).astype(np.uint8)
            clean_img_uint8 = (np.clip(clean_img_hwc, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(adv_img_uint8).save(output_dir / f"adv_{image_paths[i].name}")
            
            clean_res = yolo_model.predict(clean_img_uint8, conf=0.2, verbose=False)[0]
            adv_res = yolo_model.predict(adv_img_uint8, conf=0.2, verbose=False)[0]
            
            clean_count = len(clean_res.boxes)
            adv_count = len(adv_res.boxes)
            
            results.append({
                'image_name': image_paths[i].name,
                'clean_detections': clean_count,
                'adversarial_detections': adv_count
            })
            
            print(f"Image {i}: Clean ({clean_count}) -> Adversarial ({adv_count})")
        
        # Export results to CSV
        df = pd.DataFrame(results)
        csv_path = output_dir / "detection_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to {csv_path}")            
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback
        traceback.print_exc()