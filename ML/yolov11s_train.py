from ultralytics import YOLO

# Load YOLOv11 small (best for RTX 4050 6GB)
model = YOLO("yolo11s.pt")

model.train(
    data="pcb_yolo/data.yaml",
    imgsz=640,
    epochs=100,
    batch=4,
    device=0,
    workers=2,
    project="runs/detect",
    name="yolov11s",
    # Training optimizations
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    patience=20,         # Early stop if no improvement
    amp=True,            # Mixed precision → faster + less VRAM

    # Mild augmentations (PCB-safe)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.1,
    fliplr=0.3,
    scale=0.5,
)
