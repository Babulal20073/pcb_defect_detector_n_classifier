import os
import shutil

SRC_DIR = "/home/omen/Downloads/ML/pcb_detector/ML/pcb_classifier/dataset/pcb_data/PCB_DATASET/rotation/Spur_rotation"      # folder you are copying FROM
DEST_DIR = "dataset/pcb"       # pcb folder
PREFIX = "pcb"

os.makedirs(DEST_DIR, exist_ok=True)

# 🔹 Find current max index
existing = [
    f for f in os.listdir(DEST_DIR)
    if f.startswith(PREFIX) and f.split(".")[0].split("_")[-1].isdigit()
]

if existing:
    max_index = max(int(f.split(".")[0].split("_")[-1]) for f in existing) + 1
else:
    max_index = 0

count = 0

for file in os.listdir(SRC_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    ext = os.path.splitext(file)[1].lower()
    new_name = f"{PREFIX}_{max_index + count}{ext}"

    shutil.copy2(
        os.path.join(SRC_DIR, file),
        os.path.join(DEST_DIR, new_name)
    )
    count += 1

print(f"✅ Added {count} NEW PCB images")
