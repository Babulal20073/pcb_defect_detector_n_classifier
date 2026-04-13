from ultralytics import YOLO
import os

# ===== CONFIG =====
MODEL_PATH = "runs/classify/train2/weights/best.pt"
IMAGE_PATH = "/home/omen/Downloads/ML/pcb_detector/ML/pcb_classifier/test/test3.jpg"
THRESHOLD = 0.60
DEVICE = 0  # GPU
# ==================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    device=DEVICE,
    verbose=False
)

r = results[0]
probs = r.probs
class_names = model.names

pcb_conf = probs.data[1].item()      # index 1 = pcb
nonpcb_conf = probs.data[0].item()   # index 0 = non_pcb

print("📷 Image:", IMAGE_PATH)
print(f"🔍 Prediction: {class_names[probs.top1]}")
print(f"📊 PCB confidence: {pcb_conf:.4f}")
print(f"📊 Non-PCB confidence: {nonpcb_conf:.4f}")

if pcb_conf >= THRESHOLD:
    print("✅ RESULT: PCB detected → run defect detector")
else:
    print("❌ RESULT: Not a PCB → reject image")
