import numpy as np

from utils import image_path, output_path, read_gray, save_image, save_side_by_side


def my_hist_equalization(img: np.ndarray) -> np.ndarray:
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    cdf_min = nonzero[0]
    lut = np.round((cdf - cdf_min) * 255.0 / (img.size - cdf_min)).clip(0, 255).astype(np.uint8)
    lut[cdf == 0] = 0
    return lut[img]


def run() -> None:
    img = read_gray(image_path("runway.png"))
    equalized = my_hist_equalization(img)
    save_image(output_path("q3", "original.png"), img)
    save_image(output_path("q3", "hist_equalized.png"), equalized)
    save_side_by_side(
        [img, equalized],
        ["Original", "Manual histogram equalization"],
        output_path("q3", "comparison.png"),
    )


if __name__ == "__main__":
    run()
