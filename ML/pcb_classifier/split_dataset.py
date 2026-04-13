import os
import random
import shutil

# ========= CONFIG =========
SRC_ROOT = "dataset"          # contains pcb/ and non_pcb/
DEST_ROOT = "dataset_split"   # output folder

SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

IMG_EXTS = (".jpg", ".jpeg", ".png")
SEED = 42
# ==========================

random.seed(SEED)

classes = ["pcb", "non_pcb"]

for split in SPLIT:
    for cls in classes:
        os.makedirs(os.path.join(DEST_ROOT, split, cls), exist_ok=True)

for cls in classes:
    src_dir = os.path.join(SRC_ROOT, cls)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT["train"])
    val_end = train_end + int(total * SPLIT["val"])

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_files.items():
        for f in files:
            shutil.copy2(
                os.path.join(src_dir, f),
                os.path.join(DEST_ROOT, split, cls, f)
            )

    print(f"✅ {cls}: {total} images split into "
          f"{len(split_files['train'])} train / "
          f"{len(split_files['val'])} val / "
          f"{len(split_files['test'])} test")
