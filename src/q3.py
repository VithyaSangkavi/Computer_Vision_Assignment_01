import numpy as np
from utils import read_gray, save_image

def my_hist_equalization(img):
    hist = np.zeros(256, dtype=int)
    for pixel in img.ravel():
        hist[pixel] += 1

    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    total_pixels = img.size

    lut = ((cdf_masked - cdf_min) * 255 / (total_pixels - cdf_min)).filled(0).astype(np.uint8)
    out = lut[img]
    return out

def run():
    img = read_gray("images/runway.png")
    eq = my_hist_equalization(img)
    save_image("outputs/q3/hist_equalized.png", eq)

if __name__ == "__main__":
    run()