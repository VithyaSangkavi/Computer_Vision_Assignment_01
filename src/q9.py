import cv2
import numpy as np

from utils import image_path, output_path, read_color, save_image, save_side_by_side, write_text


def unsharp_mask(img: np.ndarray, sigma: float = 1.8, strength: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return blurred, np.clip(sharp, 0, 255).astype(np.uint8)


def run() -> None:
    img = read_color(image_path("sapphire.jpg"))
    blurred, sharpened = unsharp_mask(img)

    save_image(output_path("q9", "original.png"), img)
    save_image(output_path("q9", "blurred_for_unsharp_mask.png"), blurred)
    save_image(output_path("q9", "sharpened.png"), sharpened)
    save_side_by_side(
        [img, blurred, sharpened],
        ["Original", "Blurred", "Sharpened"],
        output_path("q9", "comparison.png"),
    )
    write_text(
        output_path("q9", "notes.txt"),
        "\n".join(
            [
                "Image used: sapphire.jpg",
                "Sharpening method: unsharp masking",
                "sigma = 1.8",
                "strength = 1.8",
                "Observed effect: edges and gem facets become more pronounced, with mild amplification of fine texture noise.",
            ]
        ),
    )


if __name__ == "__main__":
    run()
