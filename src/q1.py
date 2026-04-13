import numpy as np
from utils import read_gray, save_image, normalize_to_01, denormalize_from_01

def gamma_correction(img, gamma):
    img_norm = normalize_to_01(img)
    corrected = np.power(img_norm, gamma)
    return denormalize_from_01(corrected)

def contrast_stretch(img, r1=0.2, r2=0.8):
    img_norm = normalize_to_01(img)
    out = np.zeros_like(img_norm)

    out[img_norm < r1] = 0
    mask = (img_norm >= r1) & (img_norm <= r2)
    out[mask] = (img_norm[mask] - r1) / (r2 - r1)
    out[img_norm > r2] = 1

    return denormalize_from_01(out)

def run():
    img = read_gray("images/runway.png")

    gamma05 = gamma_correction(img, 0.5)
    gamma2 = gamma_correction(img, 2.0)
    stretch = contrast_stretch(img, 0.2, 0.8)

    save_image("outputs/q1/gamma_0_5.png", gamma05)
    save_image("outputs/q1/gamma_2.png", gamma2)
    save_image("outputs/q1/contrast_stretch.png", stretch)

if __name__ == "__main__":
    run()