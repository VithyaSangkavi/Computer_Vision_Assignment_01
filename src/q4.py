import cv2
import numpy as np
from utils import read_color, save_image

def equalize_foreground_only(gray, mask):
    fg_pixels = gray[mask > 0]
    hist = np.bincount(fg_pixels, minlength=256)
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()

    lut = ((cdf_masked - cdf_min) * 255 / (fg_pixels.size - cdf_min)).filled(0).astype(np.uint8)

    result = gray.copy()
    result[mask > 0] = lut[result[mask > 0]]
    return result

def run():
    img = read_color("images/woman_door.png")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    eq_fg = equalize_foreground_only(gray, mask)

    save_image("outputs/q4/gray.png", gray)
    save_image("outputs/q4/otsu_mask.png", mask)
    save_image("outputs/q4/equalized_foreground.png", eq_fg)

    print("Otsu threshold:", threshold)

if __name__ == "__main__":
    run()