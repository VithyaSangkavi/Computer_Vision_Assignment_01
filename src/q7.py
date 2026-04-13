import os
import cv2
import numpy as np
from utils import save_image


OUTPUT_DIR = "outputs/q7"


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_nearest(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize image using nearest-neighbor interpolation.
    """
    if scale <= 0 or scale > 10:
        raise ValueError("Scale must be in the range (0, 10].")

    h, w = image.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if image.ndim == 2:
        output = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            src_y = min(int(round(i / scale)), h - 1)
            src_x = min(int(round(j / scale)), w - 1)
            output[i, j] = image[src_y, src_x]

    return output


def resize_bilinear(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize image using bilinear interpolation.
    """
    if scale <= 0 or scale > 10:
        raise ValueError("Scale must be in the range (0, 10].")

    h, w = image.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if image.ndim == 2:
        output = np.zeros((new_h, new_w), dtype=np.float64)
    else:
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=np.float64)

    for i in range(new_h):
        for j in range(new_w):
            y = i / scale
            x = j / scale

            y1 = int(np.floor(y))
            x1 = int(np.floor(x))
            y2 = min(y1 + 1, h - 1)
            x2 = min(x1 + 1, w - 1)

            dy = y - y1
            dx = x - x1

            top = (1 - dx) * image[y1, x1] + dx * image[y1, x2]
            bottom = (1 - dx) * image[y2, x1] + dx * image[y2, x2]
            output[i, j] = (1 - dy) * top + dy * bottom

    return np.clip(output, 0, 255).astype(image.dtype)


def normalized_ssd(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute normalized sum of squared differences.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape for SSD.")

    diff = image1.astype(np.float64) - image2.astype(np.float64)
    return float(np.sum(diff ** 2) / image1.size)


def resize_to_match(image: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    If dimensions differ slightly due to rounding, resize with OpenCV just to align size
    for SSD comparison. This is optional but practical.
    """
    target_h, target_w = target_shape[:2]
    if image.ndim == 2:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def run(small_image_path: str, large_image_path: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    small = read_image(small_image_path)
    large = read_image(large_image_path)

    small_h, small_w = small.shape[:2]
    large_h, large_w = large.shape[:2]

    scale_h = large_h / small_h
    scale_w = large_w / small_w

    if not np.isclose(scale_h, scale_w, atol=1e-6):
        raise ValueError("Input images do not have a uniform scaling ratio.")

    scale = scale_h

    nearest = resize_nearest(small, scale)
    bilinear = resize_bilinear(small, scale)

    if nearest.shape != large.shape:
        nearest = resize_to_match(nearest, large.shape)

    if bilinear.shape != large.shape:
        bilinear = resize_to_match(bilinear, large.shape)

    nearest_ssd = normalized_ssd(nearest, large)
    bilinear_ssd = normalized_ssd(bilinear, large)

    save_image(os.path.join(OUTPUT_DIR, "nearest_upscaled.png"), nearest)
    save_image(os.path.join(OUTPUT_DIR, "bilinear_upscaled.png"), bilinear)

    with open(os.path.join(OUTPUT_DIR, "ssd_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Scale factor: {scale}\n")
        f.write(f"Nearest-neighbor normalized SSD: {nearest_ssd}\n")
        f.write(f"Bilinear normalized SSD: {bilinear_ssd}\n")

    print("Q7 completed.")
    print(f"Scale factor: {scale}")
    print(f"Nearest-neighbor normalized SSD: {nearest_ssd}")
    print(f"Bilinear normalized SSD: {bilinear_ssd}")


if __name__ == "__main__":
    run("images/small_image.png", "images/large_image.png")