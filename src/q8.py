import cv2

from utils import image_path, output_path, read_gray, save_image, save_side_by_side


def run() -> None:
    img = read_gray(image_path("noisy.png"))

    gaussian = cv2.GaussianBlur(img, (5, 5), 1.5)
    median = cv2.medianBlur(img, 5)

    save_image(output_path("q8", "gaussian_smoothing.png"), gaussian)
    save_image(output_path("q8", "median_filtering.png"), median)
    save_side_by_side(
        [img, gaussian, median],
        ["Original noisy image", "Gaussian smoothing", "Median filtering"],
        output_path("q8", "comparison.png"),
    )


if __name__ == "__main__":
    run()
