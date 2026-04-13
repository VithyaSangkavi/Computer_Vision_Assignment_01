def bilateral_filter_manual(img, diameter, sigma_s, sigma_r):
    h, w = img.shape
    r = diameter // 2
    padded = np.pad(img, ((r, r), (r, r)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            center = padded[i + r, j + r]
            region = padded[i:i + diameter, j:j + diameter]

            x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
            spatial = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
            range_w = np.exp(-((region - center)**2) / (2 * sigma_r**2))

            weights = spatial * range_w
            weights /= weights.sum()

            out[i, j] = np.sum(weights * region)

    return np.clip(out, 0, 255).astype(np.uint8)