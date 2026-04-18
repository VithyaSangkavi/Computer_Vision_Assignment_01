import cv2
import numpy as np

from utils import image_path, output_path, read_gray, save_image, save_side_by_side, write_text
import matplotlib.pyplot as plt


def derivative_of_gaussian_kernels(size: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2)))
    gx = -(xx / sigma**2) * gaussian
    gy = -(yy / sigma**2) * gaussian
    gx /= np.sum(np.abs(gx))
    gy /= np.sum(np.abs(gy))
    return gx, gy


def convolve_float(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padded = np.pad(image.astype(np.float64), ((radius, radius), (radius, radius)), mode="reflect")
    result = np.zeros_like(image, dtype=np.float64)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            result[row, col] = np.sum(region * kernel)
    return result


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    scaled = np.abs(image)
    if scaled.max() == 0:
        return np.zeros_like(scaled, dtype=np.uint8)
    return np.clip((scaled / scaled.max()) * 255.0, 0, 255).astype(np.uint8)


def plot_kernel_surface(kernel: np.ndarray, title: str, save_path: str) -> None:
    y = np.arange(kernel.shape[0])
    x = np.arange(kernel.shape[1])
    xx, yy = np.meshgrid(x, y)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, kernel, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def run() -> None:
    image = read_gray(image_path("brain_proton_density_slice.png"))
    gx_5, gy_5 = derivative_of_gaussian_kernels(5, 2.0)
    gx_51, _ = derivative_of_gaussian_kernels(51, 2.0)

    grad_x = convolve_float(image, gx_5)
    grad_y = convolve_float(image, gy_5)
    magnitude = np.hypot(grad_x, grad_y)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.hypot(sobel_x, sobel_y)

    save_image(output_path("q6", "original.png"), image)
    save_image(output_path("q6", "manual_grad_x.png"), normalize_for_display(grad_x))
    save_image(output_path("q6", "manual_grad_y.png"), normalize_for_display(grad_y))
    save_image(output_path("q6", "manual_gradient_magnitude.png"), normalize_for_display(magnitude))
    save_image(output_path("q6", "sobel_x.png"), normalize_for_display(sobel_x))
    save_image(output_path("q6", "sobel_y.png"), normalize_for_display(sobel_y))
    save_image(output_path("q6", "sobel_gradient_magnitude.png"), normalize_for_display(sobel_magnitude))
    plot_kernel_surface(
        gx_51,
        "51x51 derivative-of-Gaussian kernel in x, sigma=2",
        str(output_path("q6", "dog_kernel_x_51x51_surface.png")),
    )
    save_side_by_side(
        [normalize_for_display(magnitude), normalize_for_display(sobel_magnitude)],
        ["Manual DoG magnitude", "Sobel magnitude"],
        output_path("q6", "comparison.png"),
    )
    write_text(
        output_path("q6", "dog_kernel_5x5_sigma2.txt"),
        "\n\n".join(
            [
                "Gx:\n" + np.array2string(gx_5, precision=8, suppress_small=True),
                "Gy:\n" + np.array2string(gy_5, precision=8, suppress_small=True),
                "Comparison note:\nThe derivative-of-Gaussian response is smoother because differentiation is preceded by Gaussian smoothing. Sobel produces stronger, slightly sharper edge responses and is usually more sensitive to noise.",
            ]
        ),
    )


if __name__ == "__main__":
    run()
