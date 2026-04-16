import cv2
import numpy as np

from utils import image_path, output_path, read_color, save_image, save_side_by_side, write_text


def homomorphic_filter(gray: np.ndarray, sigma: float = 35.0, gamma_l: float = 0.6, gamma_h: float = 1.7) -> np.ndarray:
    gray_float = gray.astype(np.float32) + 1.0
    log_image = np.log(gray_float)

    rows, cols = gray.shape
    y = np.arange(rows) - rows / 2.0
    x = np.arange(cols) - cols / 2.0
    xx, yy = np.meshgrid(x, y)
    d2 = xx**2 + yy**2

    high_pass = 1.0 - np.exp(-d2 / (2.0 * sigma**2))
    filter_response = gamma_l + (gamma_h - gamma_l) * high_pass

    shifted = np.fft.fftshift(np.fft.fft2(log_image))
    filtered = filter_response * shifted
    restored = np.fft.ifft2(np.fft.ifftshift(filtered))

    exp_image = np.exp(np.real(restored)) - 1.0
    exp_image -= exp_image.min()
    exp_image /= exp_image.max() + 1e-8
    return (exp_image * 255.0).astype(np.uint8)


def run() -> None:
    color = read_color(image_path("highlights_and_shadows.jpg"))
    lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    corrected_l = homomorphic_filter(l_channel)
    corrected_lab = cv2.merge([corrected_l, a_channel, b_channel])
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)

    save_image(output_path("q12", "original.png"), color)
    save_image(output_path("q12", "homomorphic_corrected.png"), corrected_rgb)
    save_side_by_side(
        [color, corrected_rgb],
        ["Original", "Homomorphic filtering"],
        output_path("q12", "comparison.png"),
    )
    write_text(
        output_path("q12", "answers.txt"),
        "\n".join(
            [
                "Q12(a) Significance of the multiplicative model:",
                "The observed image is treated as illumination multiplied by reflectance. Illumination changes slowly across the scene, while reflectance carries object detail and edges.",
                "",
                "Q12(b) Log transform:",
                "Taking the logarithm converts multiplication into addition: log f = log i + log r. This makes it easier to attenuate low-frequency illumination and preserve or enhance higher-frequency reflectance terms in the frequency domain.",
                "",
                "Q12(c) Algorithm:",
                "1. Convert the image to grayscale or to a luminance channel.",
                "2. Add a small constant and take the logarithm.",
                "3. Compute the FFT and shift the zero frequency to the center.",
                "4. Apply a high-frequency emphasis filter H(u,v).",
                "5. Compute the inverse FFT.",
                "6. Exponentiate the result and normalize it back to display range.",
                "",
                "Q12(d) Comparison with histogram equalization:",
                "Histogram equalization remaps intensities globally and does not explicitly model illumination. Homomorphic filtering is preferred when the main issue is spatially varying lighting, shadows, or shading rather than only poor global contrast.",
                "",
                "Q12(e) Observation on the chosen image:",
                "The method brightens darker regions, compresses over-strong illumination variations, and improves local contrast. Mild halos or unnatural contrast can appear if the high-frequency emphasis is too strong.",
            ]
        ),
    )


if __name__ == "__main__":
    run()
