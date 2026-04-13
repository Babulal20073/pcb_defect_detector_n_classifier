from ultralytics import YOLO
import os

# =====================
# CONFIG
# =====================
DATA = "pcb_yolo/data.yaml"
DEVICE = 0
BASE_MODEL = "yolov8s.pt"

IMG = 640
BATCH = 8

FREEZE_EPOCHS = 30
FINETUNE_EPOCHS = 120

NAME_FREEZE = "exp1_freeze"
NAME_FINETUNE = "exp1_finetune"


# =====================
# EXPERIMENT RUNNER
# =====================
def run_experiment():

   
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA,
        epochs=FREEZE_EPOCHS,
        imgsz=IMG,
        batch=BATCH,
        device=DEVICE,
        name=NAME_FREEZE,
        freeze=10,               # freeze backbone layers

        # Safe augmentations
        augment=True,
        mosaic=1.0,
        mixup=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.2,
        translate=0.1,
        scale=0.5,

        # Optimization
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0005,
        patience=30,
    )

    # ------------------------------------------------
    # FIND BEST CHECKPOINT FROM FREEZE PHASE
    # ------------------------------------------------
    ckpt_best = f"runs/detect/{NAME_FREEZE}/weights/best.pt"
    ckpt_last = f"runs/detect/{NAME_FREEZE}/weights/last.pt"

    if os.path.exists(ckpt_best):
        ckpt = ckpt_best
    else:
        ckpt = ckpt_last

   

    model = YOLO(ckpt)

    model.train(
        data=DATA,
        epochs=FINETUNE_EPOCHS,
        imgsz=IMG,
        batch=max(2, BATCH // 2),
        device=DEVICE,
        name=NAME_FINETUNE,
        freeze=False,          # unfreeze!
        resume=False,          # <<< IMPORTANT FIX (avoid YOLO assertion error)

        # Data augmentations
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        fliplr=0.5,
        scale=0.5,

        # Optimizer for finetuning
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        weight_decay=0.001,

        patience=60,
    )


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    run_experiment()
