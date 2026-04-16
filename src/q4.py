import cv2
import numpy as np

from utils import image_path, output_path, read_color, save_image, save_side_by_side, write_text


def equalize_foreground_only(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    foreground = gray[mask > 0]
    hist = np.bincount(foreground, minlength=256)
    cdf = hist.cumsum()
    cdf_nonzero = cdf[cdf > 0]
    cdf_min = cdf_nonzero[0]
    lut = np.round((cdf - cdf_min) * 255.0 / (foreground.size - cdf_min)).clip(0, 255).astype(np.uint8)
    lut[cdf == 0] = 0

    result = gray.copy()
    result[mask > 0] = lut[result[mask > 0]]
    return result


def run() -> None:
    img = read_color(image_path("woman_door.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equalized = equalize_foreground_only(gray, mask)

    save_image(output_path("q4", "original_gray.png"), gray)
    save_image(output_path("q4", "otsu_mask.png"), mask)
    save_image(output_path("q4", "equalized_foreground.png"), equalized)
    save_side_by_side(
        [gray, mask, equalized],
        ["Grayscale", "Otsu foreground mask", "Foreground equalized"],
        output_path("q4", "comparison.png"),
    )
    write_text(
        output_path("q4", "notes.txt"),
        "\n".join(
            [
                f"Otsu threshold: {threshold:.2f}",
                "Revealed features: darker interior details in the room and subtle structures around the doorway become more visible after equalizing only the foreground region.",
            ]
        ),
    )


if __name__ == "__main__":
    run()
