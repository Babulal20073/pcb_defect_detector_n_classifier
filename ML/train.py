from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # fastest for training in 1 hour

model.train(
    data="data.yaml",
    epochs=40,
    imgsz=640,
    batch=8,      # adjust if RAM low
    workers=4,
    optimizer="Adam",
)
