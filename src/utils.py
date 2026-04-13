import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def read_color(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(img.shape) == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
    else:
        cv2.imwrite(path, img)

def show_histogram(img, title, save_path=None):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def normalize_to_01(img):
    return img.astype(np.float32) / 255.0

def denormalize_from_01(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)