from ultralytics import YOLO

model = YOLO("/home/omen/Downloads/ML/pcb_detector/runs/detect/train8/weights/best.pt")

metrics = model.val(data="pcb_yolo/data.yaml", imgsz=640)

print(metrics)
res = model.val(data="pcb_yolo/data.yaml")

print("\n=== MODEL PERFORMANCE ===")
print(f"Precision:       {res.results_dict['metrics/precision(B)']*100:.2f}%")
print(f"Recall:          {res.results_dict['metrics/recall(B)']*100:.2f}%")
print(f"mAP@50:          {res.results_dict['metrics/mAP50(B)']*100:.2f}%")
print(f"mAP@50-95:       {res.results_dict['metrics/mAP50-95(B)']*100:.2f}%")
