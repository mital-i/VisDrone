cd /workspace/PhantomView/VisDrone

python scripts/train_yolov8.py --model weights/best.pt --data data/visdrone.yaml --epochs 150 \
        --batch 4 --imgsz 1280 --project /workspace/PhantomView/VisDrone/runs/detect

python scripts/yolov8_video_inference.py  --input data/VisDrone2019-VID-val/sequences/uav0000086_00000_v  \
    --weights /workspace/PhantomView/VisDrone/runs/detect/visdrone/weights/best.pt --output-dir results

# ------------------------------------------------------------------------------------------- #

# cd /workspace/PhantomView/VisDrone

# # Convert train set
# python scripts/convert_annotations.py --format yolo \
#     --image-dir data/VisDrone2019-DET-train/images \
#     --annotation-dir data/VisDrone2019-DET-train/annotations \
#     --output-dir data/labels/train

# # Convert val set
# python scripts/convert_annotations.py --format yolo \
#     --image-dir data/VisDrone2019-DET-val/images \
#     --annotation-dir data/VisDrone2019-DET-val/annotations \
#     --output-dir data/labels/val
