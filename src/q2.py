import cv2
import numpy as np

from utils import image_path, output_path, read_color, save_image, save_side_by_side, show_histogram, write_text


def gamma_on_l_channel(rgb_img: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_norm = l_channel.astype(np.float32) / 255.0
    corrected_l = np.power(l_norm, gamma)
    corrected_l = np.clip(corrected_l * 255.0, 0, 255).astype(np.uint8)
    corrected_lab = cv2.merge([corrected_l, a_channel, b_channel])
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
    return l_channel, corrected_l, corrected_rgb


def run() -> None:
    gamma = 0.7
    img = read_color(image_path("highlights_and_shadows.jpg"))
    original_l, corrected_l, corrected_rgb = gamma_on_l_channel(img, gamma)

    save_image(output_path("q2", "original.png"), img)
    save_image(output_path("q2", "corrected_lab_gamma.png"), corrected_rgb)
    save_image(output_path("q2", "original_l.png"), original_l)
    save_image(output_path("q2", "corrected_l.png"), corrected_l)
    show_histogram(original_l, "Original L-channel histogram", output_path("q2", "original_hist.png"))
    show_histogram(corrected_l, "Gamma-corrected L-channel histogram", output_path("q2", "corrected_hist.png"))
    save_side_by_side(
        [img, corrected_rgb],
        ["Original", "Gamma correction on L*"],
        output_path("q2", "comparison.png"),
    )
    write_text(output_path("q2", "gamma_value.txt"), "Gamma used on the L* channel: 0.7\n")


if __name__ == "__main__":
    run()
