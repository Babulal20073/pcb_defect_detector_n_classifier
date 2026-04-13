import os
import cv2

PCB_DIR = "dataset/pcb"
IMG_SIZE = 224

count = 0
skipped = 0
already_ok = 0

for file in os.listdir(PCB_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(PCB_DIR, file)
    img = cv2.imread(path)

    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]
    if h == IMG_SIZE and w == IMG_SIZE:
        already_ok += 1
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(path, img)
    count += 1

print(f"✅ Resized {count} images")
print(f"✔ Already correct size: {already_ok}")
print(f"⚠️ Skipped unreadable: {skipped}")
