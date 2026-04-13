from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="pcb_yolo/data.yaml",
    imgsz=640,
    epochs=100,           # YOLO-M benefits from slightly more training
    batch=4,              # Fits on RTX 4050
    device=0,
    workers=2,
    lr0=0.001,            # Better stability for medium models
    optimizer="AdamW",    # Gives stronger convergence than SGD
    patience=20,          # Early stopping – avoids overfitting
    close_mosaic=10,      # Stabilize validation late in training
    cos_lr=True,          # Cosine LR schedule (recommended)
    amp=True              # Mixed precision = faster + less VRAM
)
