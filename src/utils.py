import os
from pathlib import Path

import cv2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MPLCONFIG_DIR = OUTPUTS_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def image_path(name: str) -> Path:
    return IMAGES_DIR / name


def output_path(*parts: str) -> Path:
    path = OUTPUTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def read_color(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_to_01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def denormalize_from_01(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def show_histogram(img: np.ndarray, title: str, save_path: str | Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(img.ravel(), bins=256, range=(0, 256), color="steelblue")
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=180, bbox_inches="tight")
    plt.close()


def save_side_by_side(
    images: list[np.ndarray], titles: list[str], save_path: str | Path, cmap: str | None = None
) -> None:
    cols = len(images)
    plt.figure(figsize=(5 * cols, 4))
    for index, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, cols, index)
        if image.ndim == 2:
            plt.imshow(image, cmap=cmap or "gray")
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=180, bbox_inches="tight")
    plt.close()
