import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import image_path, output_path, read_gray, save_image, save_side_by_side, write_text


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2)))
    kernel /= kernel.sum()
    return kernel


def convolve_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padded = np.pad(image.astype(np.float64), ((radius, radius), (radius, radius)), mode="reflect")
    result = np.zeros_like(image, dtype=np.float64)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            result[row, col] = np.sum(region * kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def plot_kernel_surface(kernel: np.ndarray, title: str, save_path: str) -> None:
    y = np.arange(kernel.shape[0])
    x = np.arange(kernel.shape[1])
    xx, yy = np.meshgrid(x, y)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, kernel, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def run() -> None:
    image = read_gray(image_path("rice.png"))
    kernel_5 = gaussian_kernel(5, 2.0)
    kernel_51 = gaussian_kernel(51, 2.0)
    manual = convolve_manual(image, kernel_5)
    opencv = cv2.GaussianBlur(image, (5, 5), sigmaX=2.0, sigmaY=2.0)

    save_image(output_path("q5", "original.png"), image)
    save_image(output_path("q5", "manual_gaussian_blur.png"), manual)
    save_image(output_path("q5", "opencv_gaussian_blur.png"), opencv)
    plot_kernel_surface(kernel_51, "51x51 Gaussian kernel, sigma=2", str(output_path("q5", "gaussian_kernel_51x51_surface.png")))
    save_side_by_side(
        [image, manual, opencv],
        ["Original", "Manual Gaussian smoothing", "OpenCV Gaussian smoothing"],
        output_path("q5", "comparison.png"),
    )
    write_text(
        output_path("q5", "gaussian_kernel_5x5_sigma2.txt"),
        np.array2string(kernel_5, precision=8, suppress_small=True),
    )


if __name__ == "__main__":
    run()
