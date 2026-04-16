import cv2
import numpy as np

from utils import image_path, output_path, read_gray, save_image, save_side_by_side, write_text


def bilateral_filter_manual(img: np.ndarray, diameter: int, sigma_s: float, sigma_r: float) -> np.ndarray:
    if diameter % 2 == 0:
        raise ValueError("Kernel diameter must be odd.")

    radius = diameter // 2
    padded = np.pad(img.astype(np.float64), ((radius, radius), (radius, radius)), mode="reflect")
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_s**2))

    out = np.zeros_like(img, dtype=np.float64)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            region = padded[row : row + diameter, col : col + diameter]
            center = padded[row + radius, col + radius]
            range_weights = np.exp(-((region - center) ** 2) / (2.0 * sigma_r**2))
            weights = spatial * range_weights
            weights /= weights.sum()
            out[row, col] = np.sum(weights * region)

    return np.clip(out, 0, 255).astype(np.uint8)


def run() -> None:
    img = read_gray(image_path("einstein.png"))

    gaussian = cv2.GaussianBlur(img, (5, 5), 1.5)
    bilateral_cv = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=7)
    bilateral_manual = bilateral_filter_manual(img, diameter=7, sigma_s=7.0, sigma_r=35.0)

    save_image(output_path("q10", "original.png"), img)
    save_image(output_path("q10", "gaussian_blur.png"), gaussian)
    save_image(output_path("q10", "bilateral_opencv.png"), bilateral_cv)
    save_image(output_path("q10", "bilateral_manual.png"), bilateral_manual)
    save_side_by_side(
        [img, gaussian, bilateral_cv, bilateral_manual],
        ["Original", "Gaussian blur", "OpenCV bilateral", "Manual bilateral"],
        output_path("q10", "comparison.png"),
    )
    write_text(
        output_path("q10", "notes.txt"),
        "\n".join(
            [
                "Image used: einstein.png",
                "Manual bilateral parameters: diameter=7, sigma_s=7.0, sigma_r=35.0",
                "OpenCV bilateral parameters: d=7, sigmaColor=35, sigmaSpace=7",
                "Observation: bilateral filtering preserves edge transitions better than Gaussian smoothing while still smoothing small intensity variations in flat regions.",
            ]
        ),
    )


if __name__ == "__main__":
    run()
