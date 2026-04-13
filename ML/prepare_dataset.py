import os
import random
import shutil
import xml.etree.ElementTree as ET

# =========================
# CONFIG
# =========================

DATASET_ROOT = "PCB_DATASET"
ANNOT_DIR = os.path.join(DATASET_ROOT, "Annotations")
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")

OUT_DIR = "pcb_yolo"
OUT_IMG = os.path.join(OUT_DIR, "images")
OUT_LABEL = os.path.join(OUT_DIR, "labels")

# XML contains lowercase defect names
class_map = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5
}

# Train/val split
TRAIN_RATIO = 0.8


# =========================
# HELPERS
# =========================

def ensure_dirs():
    """Create YOLO folder structure."""
    for p in [OUT_DIR, OUT_IMG, OUT_LABEL]:
        os.makedirs(p, exist_ok=True)

    for subset in ["train", "val"]:
        os.makedirs(os.path.join(OUT_IMG, subset), exist_ok=True)
        os.makedirs(os.path.join(OUT_LABEL, subset), exist_ok=True)

    print("[OK] YOLO folder structure created.")


def find_image(xml_path: str):
    """
    Extract real image path from XML.
    Matches the filename by searching all images.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text.strip()
    base = os.path.basename(filename)

    # Search inside all defect folders under images/
    for defect in os.listdir(IMAGE_DIR):
        defect_path = os.path.join(IMAGE_DIR, defect)
        if os.path.isdir(defect_path):
            candidate = os.path.join(defect_path, base)
            if os.path.exists(candidate):
                return candidate

    # No match found
    return None


def convert_xml_to_yolo(xml_path, label_out_path):
    """Convert XML annotation to YOLO TXT."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = int(root.find("size").find("width").text)
    h = int(root.find("size").find("height").text)

    lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip().lower()

        if cls_name not in class_map:
            print(f"[ERROR] Unknown class '{cls_name}' in {xml_path}")
            continue

        cls_id = class_map[cls_name]

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    with open(label_out_path, "w") as f:
        f.write("\n".join(lines))


def process(xml_list, subset):
    """Process XML files into YOLO format for train/val subsets."""
    img_out_dir = os.path.join(OUT_IMG, subset)
    lbl_out_dir = os.path.join(OUT_LABEL, subset)

    for xml_file in xml_list:
        img_path = find_image(xml_file)

        if img_path is None:
            print(f"[WARN] No matching image found for XML: {xml_file}")
            continue

        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        # Copy image
        shutil.copy(img_path, os.path.join(img_out_dir, img_name))

        # Convert annotation
        label_out = os.path.join(lbl_out_dir, f"{base_name}.txt")
        convert_xml_to_yolo(xml_file, label_out)


# =========================
# MAIN
# =========================

def collect_xml_files():
    """Recursively collect all XML files from all defect folders."""
    xml_files = []
    for defect in os.listdir(ANNOT_DIR):
        defect_dir = os.path.join(ANNOT_DIR, defect)
        if os.path.isdir(defect_dir):
            for f in os.listdir(defect_dir):
                if f.endswith(".xml"):
                    xml_files.append(os.path.join(defect_dir, f))
    return xml_files


if __name__ == "__main__":
    ensure_dirs()

    xml_files = collect_xml_files()
    print(f"[INFO] Total XML files: {len(xml_files)}")

    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * TRAIN_RATIO)

    train_xml = xml_files[:split_idx]
    val_xml = xml_files[split_idx:]

    print(f"[INFO] Train: {len(train_xml)}   Val: {len(val_xml)}")

    process(train_xml, "train")
    process(val_xml, "val")

    print("[DONE] Dataset prepared successfully!")
