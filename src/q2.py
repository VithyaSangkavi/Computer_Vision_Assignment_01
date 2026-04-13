import cv2
import numpy as np
from utils import read_color, save_image, show_histogram

def gamma_l_channel(rgb_img, gamma=0.7):
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    L_norm = L.astype(np.float32) / 255.0
    L_corr = np.power(L_norm, gamma)
    L_corr = np.clip(L_corr * 255, 0, 255).astype(np.uint8)

    corrected_lab = cv2.merge([L_corr, a, b])
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)

    return L, L_corr, corrected_rgb

def run():
    img = read_color("images/highlights_and_shadows.jpg")
    L_orig, L_corr, corrected = gamma_l_channel(img, gamma=0.7)

    save_image("outputs/q2/corrected_lab_gamma.png", corrected)
    show_histogram(L_orig, "Original L Histogram", "outputs/q2/original_hist.png")
    show_histogram(L_corr, "Corrected L Histogram", "outputs/q2/corrected_hist.png")

if __name__ == "__main__": 
    run()