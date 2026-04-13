from ultralytics import YOLO

model = YOLO("yolov8s.pt")  

model.train(
    data="pcb_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,            
    workers=2,
)
