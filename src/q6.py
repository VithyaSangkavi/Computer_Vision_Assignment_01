import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_gray, save_image


OUTPUT_DIR = "outputs/q6"


def gaussian_derivative_kernels(size: int = 5, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute derivative-of-Gaussian kernels in x and y directions.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

    G = (1.0 / (2 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    Gx = -(x / sigma**2) * G
    Gy = -(y / sigma**2) * G

    return Gx.astype(np.float64), Gy.astype(np.float64)


def convolve2d_float(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual convolution returning float output.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale.")

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros(image.shape, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize float image for saving/display.
    """
    abs_img = np.abs(image)
    if abs_img.max() == 0:
        return np.zeros_like(abs_img, dtype=np.uint8)
    norm = (abs_img / abs_img.max()) * 255.0
    return norm.astype(np.uint8)


def plot_kernel_surface(kernel: np.ndarray, title: str, save_path: str) -> None:
    size_y, size_x = kernel.shape
    x = np.arange(size_x)
    y = np.arange(size_y)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, kernel, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_kernel_text(kernel: np.ndarray, save_path: str) -> None:
    np.set_printoptions(precision=8, suppress=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(np.array2string(kernel, separator=", "))


def run(image_path: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = read_gray(image_path)

    # Part (b): 5x5 DoG kernels for sigma=2
    Gx_5, Gy_5 = gaussian_derivative_kernels(size=5, sigma=2.0)
    save_kernel_text(Gx_5, os.path.join(OUTPUT_DIR, "dog_kernel_x_5x5_sigma2.txt"))
    save_kernel_text(Gy_5, os.path.join(OUTPUT_DIR, "dog_kernel_y_5x5_sigma2.txt"))

    # Part (c): 51x51 DoG kernel visualization
    Gx_51, _ = gaussian_derivative_kernels(size=51, sigma=2.0)
    plot_kernel_surface(
        Gx_51,
        "51x51 Derivative of Gaussian Kernel (X direction, sigma=2)",
        os.path.join(OUTPUT_DIR, "dog_kernel_x_51x51_surface.png")
    )

    # Part (d): Apply computed DoG kernels
    grad_x = convolve2d_float(image, Gx_5)
    grad_y = convolve2d_float(image, Gy_5)

    grad_x_vis = normalize_for_display(grad_x)
    grad_y_vis = normalize_for_display(grad_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_vis = normalize_for_display(magnitude)

    save_image(os.path.join(OUTPUT_DIR, "manual_grad_x.png"), grad_x_vis)
    save_image(os.path.join(OUTPUT_DIR, "manual_grad_y.png"), grad_y_vis)
    save_image(os.path.join(OUTPUT_DIR, "manual_gradient_magnitude.png"), magnitude_vis)

    # Part (e): Sobel gradients using OpenCV
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    save_image(os.path.join(OUTPUT_DIR, "sobel_x.png"), normalize_for_display(sobel_x))
    save_image(os.path.join(OUTPUT_DIR, "sobel_y.png"), normalize_for_display(sobel_y))
    save_image(os.path.join(OUTPUT_DIR, "sobel_gradient_magnitude.png"), normalize_for_display(sobel_magnitude))

    print("Q6 completed.")
    print("5x5 DoG X kernel:\n", Gx_5)
    print("5x5 DoG Y kernel:\n", Gy_5)


if __name__ == "__main__":
    run("images/runway.png")