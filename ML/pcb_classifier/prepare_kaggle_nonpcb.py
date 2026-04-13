import cv2
import os
import random
import shutil

SRC_DIR = "/home/omen/Downloads/ML/pcb_detector/ML/pcb_classifier/archive/data"   # ← path where Kaggle images are
OUT_DIR = "dataset/non_pcb"

os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 224
NUM_SAMPLES = 1000

images = os.listdir(SRC_DIR)
random.shuffle(images)

selected = images[:NUM_SAMPLES]

for idx, img_name in enumerate(selected):
    img_path = os.path.join(SRC_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(f"{OUT_DIR}/kaggle_nonpcb_{idx}.jpg", img)

print("✅ Added 1000 Kaggle non-PCB images")
