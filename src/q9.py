def unsharp_mask(img, sigma=1.5, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return sharp