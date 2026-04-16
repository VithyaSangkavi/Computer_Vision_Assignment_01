from pathlib import Path

import cv2
import numpy as np

from utils import output_path, read_color, save_image, write_text


def resize_nearest(image: np.ndarray, scale: float) -> np.ndarray:
    new_height = max(1, int(round(image.shape[0] * scale)))
    new_width = max(1, int(round(image.shape[1] * scale)))
    out = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    for row in range(new_height):
        for col in range(new_width):
            src_row = min(int(round(row / scale)), image.shape[0] - 1)
            src_col = min(int(round(col / scale)), image.shape[1] - 1)
            out[row, col] = image[src_row, src_col]
    return out


def resize_bilinear(image: np.ndarray, scale: float) -> np.ndarray:
    new_height = max(1, int(round(image.shape[0] * scale)))
    new_width = max(1, int(round(image.shape[1] * scale)))
    out = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float64)

    for row in range(new_height):
        for col in range(new_width):
            src_y = row / scale
            src_x = col / scale

            y1 = int(np.floor(src_y))
            x1 = int(np.floor(src_x))
            y2 = min(y1 + 1, image.shape[0] - 1)
            x2 = min(x1 + 1, image.shape[1] - 1)

            dy = src_y - y1
            dx = src_x - x1

            top = (1 - dx) * image[y1, x1] + dx * image[y1, x2]
            bottom = (1 - dx) * image[y2, x1] + dx * image[y2, x2]
            out[row, col] = (1 - dy) * top + dy * bottom

    return np.clip(out, 0, 255).astype(np.uint8)


def normalized_ssd(image_a: np.ndarray, image_b: np.ndarray) -> float:
    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    return float(np.mean(diff**2))


def resize_to_shape_nearest(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    out = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    scale_y = target_height / image.shape[0]
    scale_x = target_width / image.shape[1]

    for row in range(target_height):
        for col in range(target_width):
            src_row = min(int(round(row / scale_y)), image.shape[0] - 1)
            src_col = min(int(round(col / scale_x)), image.shape[1] - 1)
            out[row, col] = image[src_row, src_col]

    return out


def resize_to_shape_bilinear(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    out = np.zeros((target_height, target_width, image.shape[2]), dtype=np.float64)
    scale_y = target_height / image.shape[0]
    scale_x = target_width / image.shape[1]

    for row in range(target_height):
        for col in range(target_width):
            src_y = row / scale_y
            src_x = col / scale_x

            y1 = int(np.floor(src_y))
            x1 = int(np.floor(src_x))
            y2 = min(y1 + 1, image.shape[0] - 1)
            x2 = min(x1 + 1, image.shape[1] - 1)

            dy = src_y - y1
            dx = src_x - x1

            top = (1 - dx) * image[y1, x1] + dx * image[y1, x2]
            bottom = (1 - dx) * image[y2, x1] + dx * image[y2, x2]
            out[row, col] = (1 - dy) * top + dy * bottom

    return np.clip(out, 0, 255).astype(np.uint8)


def discover_pairs() -> list[tuple[str, Path, Path]]:
    q7_dir = Path(__file__).resolve().parents[1] / "images" / "q7_images"
    if not q7_dir.exists():
        raise FileNotFoundError("Q7 image folder not found: images/q7_images")

    pairs: list[tuple[str, Path, Path]] = []
    used_smalls: set[Path] = set()

    for large_path in sorted(q7_dir.iterdir()):
        if not large_path.is_file():
            continue

        stem = large_path.stem.lower()
        if "small" in stem:
            continue

        candidates = [
            q7_dir / f"{large_path.stem}small{large_path.suffix}",
            q7_dir / f"{large_path.stem}_small{large_path.suffix}",
            q7_dir / f"{large_path.stem}Small{large_path.suffix}",
            q7_dir / f"{large_path.stem}_Small{large_path.suffix}",
        ]

        for candidate in candidates:
            if candidate.exists():
                pairs.append((large_path.stem, large_path, candidate))
                used_smalls.add(candidate)
                break

    if not pairs:
        raise ValueError("No Q7 large/small image pairs were found in images/q7_images.")

    return pairs


def run() -> None:
    pairs = discover_pairs()
    lines = ["Q7 results using the provided large/small image pairs in images/q7_images."]

    for label, large_path, small_path in pairs:
        original = read_color(large_path)
        small = read_color(small_path)
        scale_y = original.shape[0] / small.shape[0]
        scale_x = original.shape[1] / small.shape[1]
        if np.isclose(scale_x, scale_y, atol=1e-3):
            upscale_factor = (scale_x + scale_y) / 2.0
            nearest = resize_nearest(small, upscale_factor)
            bilinear = resize_bilinear(small, upscale_factor)
            if nearest.shape != original.shape:
                nearest = cv2.resize(nearest, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
            if bilinear.shape != original.shape:
                bilinear = cv2.resize(bilinear, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            nearest = resize_to_shape_nearest(small, original.shape[0], original.shape[1])
            bilinear = resize_to_shape_bilinear(small, original.shape[0], original.shape[1])
            upscale_factor = float("nan")

        nearest_ssd = normalized_ssd(nearest, original)
        bilinear_ssd = normalized_ssd(bilinear, original)

        save_image(output_path("q7", f"{label}_original.png"), original)
        save_image(output_path("q7", f"{label}_small.png"), small)
        save_image(output_path("q7", f"{label}_nearest.png"), nearest)
        save_image(output_path("q7", f"{label}_bilinear.png"), bilinear)

        lines.extend(
            [
                "",
                f"Large image: {large_path.name}",
                f"Small image: {small_path.name}",
                f"Horizontal scale factor: {scale_x:.6f}",
                f"Vertical scale factor: {scale_y:.6f}",
                (
                    f"Uniform upscale factor used in manual zoom: {upscale_factor:.6f}"
                    if not np.isnan(upscale_factor)
                    else "Uniform upscale factor used in manual zoom: not used due to minor saved-image rounding mismatch."
                ),
                f"Nearest-neighbor normalized SSD: {nearest_ssd:.6f}",
                f"Bilinear normalized SSD: {bilinear_ssd:.6f}",
            ]
        )

    write_text(output_path("q7", "ssd_results.txt"), "\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
