# PhantomView -- VisDrone Pipeline

## Dataset Setup

### 1. Download VisDrone Dataset 

Download the VisDrone2019 dataset (Object Detection in Images: trainset and valset + Object Detection in Videos: trainset and valset + Multi-Object Tracking: trainset and valset) from the [official repository](https://github.com/VisDrone/VisDrone-Dataset), then place the downloaded .zip files in the 'data' folder and extract them as following: 

```bash
cd /path/to/PhantomView/VisDrone/data

# Extract
unzip VisDrone2019-VID-train.zip
unzip VisDrone2019-VID-val.zip
unzip VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-val.zip
unzip VisDrone2019-MOT-train.zip
unzip VisDrone2019-MOT-val.zip
```

### 2. Convert Annotations to YOLO Format

```bash
cd /path/to/PhantomView/VisDrone

# Convert training set
python3 scripts/convert_annotations.py --format yolo \
    --image-dir data/VisDrone2019-DET-train/images \
    --annotation-dir data/VisDrone2019-DET-train/annotations \
    --output-dir data/VisDrone2019-DET-train/labels

# Convert validation set
python3 scripts/convert_annotations.py --format yolo \
    --image-dir data/VisDrone2019-DET-val/images \
    --annotation-dir data/VisDrone2019-DET-val/annotations \
    --output-dir data/VisDrone2019-DET-val/labels
```

### 3. Verify Dataset Structure

Your data directory should look like:
```
VisDrone/data/
├── visdrone.yaml
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── labels/
    └── annotations/
└── VisDrone2019-VID-train/
    └── sequences/
└── VisDrone2019-VID-val/
    └── sequences/
└── VisDrone2019-MOT-train/
    ├── sequences/
    └── annotations/
└── VisDrone2019-MOT-val/
    ├── sequences/
    └── annotations/

```

## Training

### Basic Training 

Note: You can skip this step. No need to re-run the training as we already have the pretrained weights

```bash
python3 scripts/train_yolov8.py --model yolov8n.pt --data data/visdrone.yaml --epochs 100
```

### Training with Higher Resolution (Recommended for Small Objects)

Note: You can skip this step. No need to re-run the training as we already have the pretrained weights

```bash
python3 scripts/train_yolov8.py --model yolov8n.pt --data data/visdrone.yaml --epochs 150 \
    --batch 4 --imgsz 1280 --project runs/detect
```

### Fine-tuning from a Checkpoint

Note: You can skip this step. No need to re-run the training as we already have the pretrained weights

```bash
python3 scripts/train_yolov8.py --model weights/best.pt --data data/visdrone.yaml --epochs 150 \
    --batch 4 --imgsz 1280 --project runs/detect
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to model weights (.pt) |
| `--data` | data/visdrone.yaml | Dataset configuration file |
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Image size (use 1280 for small objects) |
| `--device` | auto | cuda, cpu, or GPU id |
| `--cache` | False | Cache images for faster training |

### Output

Trained weights are saved to:
```
VisDrone/runs/detect/visdrone/weights/
├── best.pt   # Best model (highest mAP)
└── last.pt   # Latest checkpoint
```

## Inference

Pre-trained weights are available for download: [Google Drive](https://drive.google.com/drive/folders/1mMHTzIsvV19XL_vo9y73TyJ73ptJ7tX7?usp=sharing). Download the visdrone folder then place it in the runs/detect/ folder created by the following command:

```bash
mkdir -r runs/detect/
```

### Video/Sequence Inference

```bash
cd VisDrone

# Run on a video file
python3 scripts/yolov8_video_inference.py \
    --input path/to/video.mp4 \
    --weights runs/detect/visdrone/weights/best.pt \
    --output-dir results

# Run on a frame sequence directory
python3 scripts/yolov8_video_inference.py \
    --input data/VisDrone2019-VID-val/sequences/uav0000305_00000_v \
    --weights runs/detect/visdrone/weights/best.pt \
    --output-dir results

# Process all sequences in a directory
python3 scripts/yolov8_video_inference.py \
    --input data/VisDrone2019-VID-val/sequences \
    --weights runs/detect/visdrone/weights/best.pt \
    --batch-sequences \
    --output-dir results
```

### Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | required | Path to trained weights |
| `--input` | required | Video file or sequence directory |
| `--output-dir` | yolov8_outputs | Output directory |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--imgsz` | 1280 | Inference image size (should match training) |
| `--device` | cpu | cuda:0, cpu, etc. |
| `--fps` | 30 | Output video FPS |

### Multi-Object Tracking (DeepSORT)

Add the `--track` flag to enable DeepSORT tracking on top of per-frame detections. Each detected object is assigned a persistent ID across frames.

```bash
# Track objects in a video
python3 scripts/yolov8_video_inference.py \
    --input path/to/video.mp4 \
    --weights runs/detect/visdrone/weights/best.pt \
    --output-dir results \
    --track

# Custom tracker settings
python3 scripts/yolov8_video_inference.py \
    --input data/VisDrone2019-VID-val/sequences/uav0000305_00000_v \
    --weights runs/detect/visdrone/weights/best.pt \
    --output-dir results \
    --track --track-max-age 50 --track-n-init 5

```

#### Tracking Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--track` | off | Enable DeepSORT multi-object tracking |
| `--track-max-age` | 30 | Frames to keep a lost track alive |
| `--track-n-init` | 3 | Consecutive detections to confirm a track |
| `--track-max-cosine-distance` | 0.2 | Max cosine distance for appearance matching |

When `--save-txt` is used with tracking, the output format changes to `track_id class_id x1 y1 x2 y2` (one line per tracked object).

The tracker can also be used programmatically:

```python
from visdrone_toolkit.tracker import DeepSORTTracker

tracker = DeepSORTTracker(max_age=30, n_init=3)
# Per frame: feed YOLO detections
tracks = tracker.update(frame, boxes, scores, class_ids)
annotated = tracker.draw_tracks(frame, tracks, class_names)
```

### MOT Evaluation

Evaluate DeepSORT tracking against VisDrone MOT ground truth annotations:

```bash
# Evaluate on all MOT-val sequences
python3 scripts/eval_mot.py \
    --mot-root data/VisDrone2019-MOT-val \
    --weights runs/detect/visdrone/weights/best.pt \
    --device cuda:0

# Evaluate specific sequences with custom settings
python3 scripts/eval_mot.py \
    --mot-root data/VisDrone2019-MOT-val \
    --weights runs/detect/visdrone/weights/best.pt \
    --sequences uav0000086_00000_v uav0000137_00000_v \
    --conf 0.3 --max-frames 100
```

#### Evaluation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--mot-root` | required | Path to VisDrone MOT split (e.g., MOT-val) |
| `--weights` | required | Path to trained weights |
| `--conf` | 0.25 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--imgsz` | 1280 | Inference image size |
| `--device` | cpu | Device (cuda:0, cpu, etc.) |
| `--sequences` | all | Run only on specific sequence names |
| `--max-frames` | all | Maximum frames per sequence |
| `--output-dir` | None | Save per-sequence prediction files |


## Project Structure

```
PhantomView/
├── README.md
├── requirements.txt
└── VisDrone/
    ├── data/
    │   ├── visdrone.yaml          # Dataset config
    │   ├── VisDrone2019-DET-train/
    │   └── VisDrone2019-DET-val/
    ├── scripts/
    │   ├── train_yolov8.py        # Training script
    │   ├── yolov8_video_inference.py  # Inference & tracking script
    │   ├── eval_mot.py            # MOT evaluation script
    │   └── convert_annotations.py # Annotation converter
    ├── visdrone_toolkit/          # Dataset & tracking utilities
    └── runs/                      # Training outputs
```

## VisDrone Classes

| ID | Class |
|----|-------|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

## Offline / FIPS-Compliant Operation

In earlier versions of this pipeline, the ultralytics library would silently make HTTPS requests at runtime — to check for package updates, download missing model weights, and fetch font files. On systems running a FIPS-validated OpenSSL module, these TLS calls triggered `OpenSSL is not FIPS-compliant` errors because the default OpenSSL provider loaded by Python's `ssl` module is not FIPS-approved.

To eliminate this, all scripts now set `YOLO_OFFLINE=1` **before** importing ultralytics, which disables every network request the library would otherwise make. This ensures no OpenSSL/TLS calls are triggered at runtime, which is required for FIPS 140-2/140-3 compliant environments.

**Before running any script**, ensure the following assets are available locally:

1. **Model weights** — all scripts require an explicit `--weights` / `--model` path. No weights are auto-downloaded. For inference/evaluation, use your trained weights at `runs/detect/visdrone/weights/best.pt`.

2. **Dataset** — the VisDrone data must already be extracted under `data/` (see [Dataset Setup](#1-download-visdrone-dataset) above).

3. **DeepSORT embedder** — the MobileNet weights (`mobilenetv2_bottleneck_wts.pt`) are bundled inside the `deep-sort-realtime` pip package (should be installed when running the pip install -r requirements earlier). No separate download is needed.

With these assets in place, the entire pipeline (training, inference, tracking, evaluation) runs fully offline.

To further test if the codebase is FIPS-safe, you can run the following script:

```bash
python3 scripts/test_fips_compliance.py --weights runs/detect/visdrone/weights/best.pt
```