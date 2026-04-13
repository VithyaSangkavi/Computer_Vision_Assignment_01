import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_gray, save_image


OUTPUT_DIR = "outputs/q5"


def gaussian_kernel(size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """
    Compute a normalized 2D Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    kernel = np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float64)


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution manually using reflect padding.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale.")

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def plot_kernel_surface(kernel: np.ndarray, title: str, save_path: str) -> None:
    """
    Save a 3D surface plot of a kernel.
    """
    size_y, size_x = kernel.shape
    x = np.arange(size_x)
    y = np.arange(size_y)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, kernel, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_kernel_text(kernel: np.ndarray, save_path: str) -> None:
    """
    Save kernel values to a text file.
    """
    np.set_printoptions(precision=8, suppress=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(np.array2string(kernel, separator=", "))


def run(image_path: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = read_gray(image_path)

    # Part (a): 5x5 normalized Gaussian kernel for sigma = 2
    kernel_5 = gaussian_kernel(size=5, sigma=2.0)
    save_kernel_text(kernel_5, os.path.join(OUTPUT_DIR, "gaussian_kernel_5x5_sigma2.txt"))

    # Part (b): 51x51 Gaussian kernel 3D surface plot
    kernel_51 = gaussian_kernel(size=51, sigma=2.0)
    plot_kernel_surface(
        kernel_51,
        "51x51 Gaussian Kernel Surface (sigma=2)",
        os.path.join(OUTPUT_DIR, "gaussian_kernel_51x51_surface.png")
    )

    # Part (c): Manual Gaussian smoothing
    manual_smoothed = convolve2d_manual(image, kernel_5)
    save_image(os.path.join(OUTPUT_DIR, "manual_gaussian_blur.png"), manual_smoothed)

    # Part (d): OpenCV Gaussian smoothing
    opencv_smoothed = cv2.GaussianBlur(image, (5, 5), 2.0)
    save_image(os.path.join(OUTPUT_DIR, "opencv_gaussian_blur.png"), opencv_smoothed)

    print("Q5 completed.")
    print("5x5 Gaussian kernel:\n", kernel_5)


if __name__ == "__main__":
    run("images/runway.png")