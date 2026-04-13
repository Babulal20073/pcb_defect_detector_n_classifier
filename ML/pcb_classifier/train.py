from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset_split",
    epochs=25,
    imgsz=224,
    batch=32,
    degrees=30,
    device=0
)
