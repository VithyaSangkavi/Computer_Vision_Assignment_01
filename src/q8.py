img = read_gray("images/noisy.png")
gaussian = cv2.GaussianBlur(img, (5, 5), 1.5)
median = cv2.medianBlur(img, 5)