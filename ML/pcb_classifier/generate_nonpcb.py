import cv2
import numpy as np
import os
import random

OUT_DIR = "dataset/non_pcb"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 224
NUM_IMAGES = 1000

def solid_color(color):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img[:] = color
    return img

def random_noise():
    return np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

def add_noise(img):
    noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def add_blur(img):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)

for i in range(NUM_IMAGES):
    choice = random.randint(0, 4)

    if choice == 0:
        img = solid_color((0, 255, 0))     # green
    elif choice == 1:
        img = solid_color((255, 255, 255)) # white
    elif choice == 2:
        img = solid_color((0, 0, 0))       # black
    elif choice == 3:
        img = random_noise()
    else:
        img = add_noise(random_noise())

    if random.random() > 0.5:
        img = add_blur(img)

    cv2.imwrite(f"{OUT_DIR}/gen_nonpcb_{i}.jpg", img)

print("✅ Generated 1000 synthetic non-PCB images")
