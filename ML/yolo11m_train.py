from ultralytics import YOLO

# Load YOLOv11 Medium
model = YOLO("yolo11m.pt")

model.train(
    data="pcb_yolo/data.yaml",
    imgsz=640,
    epochs=100,
    batch=4,
    device=0,
    workers=2,

    project="runs/detect",
    name="yolov11m",

    # optimizer and scheduler
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    patience=20,
    amp=True,

    # Light PCB-safe augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.3,
    flipud=0.1,
    scale=0.5,
)
