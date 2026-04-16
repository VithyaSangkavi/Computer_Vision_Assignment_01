import numpy as np

from utils import image_path, output_path, read_gray, save_image, save_side_by_side


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    normalized = img.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def contrast_stretch(img: np.ndarray, r1: float = 0.2, r2: float = 0.8) -> np.ndarray:
    normalized = img.astype(np.float32) / 255.0
    stretched = np.zeros_like(normalized)
    middle = (normalized >= r1) & (normalized <= r2)
    stretched[middle] = (normalized[middle] - r1) / (r2 - r1)
    stretched[normalized > r2] = 1.0
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def run() -> None:
    img = read_gray(image_path("runway.png"))
    gamma_05 = gamma_correction(img, 0.5)
    gamma_2 = gamma_correction(img, 2.0)
    stretched = contrast_stretch(img, 0.2, 0.8)

    save_image(output_path("q1", "original.png"), img)
    save_image(output_path("q1", "gamma_0_5.png"), gamma_05)
    save_image(output_path("q1", "gamma_2_0.png"), gamma_2)
    save_image(output_path("q1", "contrast_stretch.png"), stretched)
    save_side_by_side(
        [img, gamma_05, gamma_2, stretched],
        ["Original", "Gamma 0.5", "Gamma 2.0", "Contrast stretch"],
        output_path("q1", "comparison.png"),
    )


if __name__ == "__main__":
    run()
