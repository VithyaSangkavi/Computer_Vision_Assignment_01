# IN4640 Assignment 1 Report

**Name:** `[Your Name]`  
**Index Number:** `[Your Index Number]`  
**GitHub / SVN Profile:** `[Paste your repository profile link here]`

## Introduction

This report presents the implementation and analysis for Assignment 1 on intensity transformations and neighborhood filtering. All programs were implemented in Python using NumPy, OpenCV, and Matplotlib. The code is organized as separate scripts for each question under `src/`, and the generated results are saved under `outputs/`.

## Q1. Intensity Transformations on the Runway Image

Gamma correction and contrast stretching were applied to the runway image. Gamma correction with `gamma = 0.5` brightened darker regions and revealed shadow detail. Gamma correction with `gamma = 2.0` darkened the image and emphasized bright structures. The contrast stretching transformation expanded intensities in the mid-range `[0.2, 0.8]`, which improved visibility in regions with moderate contrast.

Important code used:

```python
def gamma_correction(img, gamma):
    normalized = img.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def contrast_stretch(img, r1=0.2, r2=0.8):
    normalized = img.astype(np.float32) / 255.0
    stretched = np.zeros_like(normalized)
    middle = (normalized >= r1) & (normalized <= r2)
    stretched[middle] = (normalized[middle] - r1) / (r2 - r1)
    stretched[normalized > r2] = 1.0
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
```

![Q1 comparison](outputs/q1/comparison.png)

The results show how nonlinear intensity mappings can selectively enhance or suppress different tonal regions. Gamma values below 1 improve visibility in dark regions, while values above 1 compress them.

## Q2. Gamma Correction in the L\*a\*b\* Color Space

Gamma correction was applied only to the `L*` channel of the image in the L\*a\*b\* color space using:

- `gamma = 0.7`

This approach changes luminance while preserving chromatic information better than applying the same transformation directly in RGB space. The corrected image appears brighter in the shadowed areas while keeping the colors visually stable.

Important code used:

```python
lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
l_channel, a_channel, b_channel = cv2.split(lab)
l_norm = l_channel.astype(np.float32) / 255.0
corrected_l = np.power(l_norm, gamma)
corrected_l = np.clip(corrected_l * 255.0, 0, 255).astype(np.uint8)
```

![Q2 comparison](outputs/q2/comparison.png)

The histograms of the original and corrected `L*` channel confirm that the transformation shifts a larger number of pixels toward higher intensity values.

![Q2 original histogram](outputs/q2/original_hist.png)
![Q2 corrected histogram](outputs/q2/corrected_hist.png)

## Q3. Manual Histogram Equalization

A custom histogram equalization function was implemented without calling OpenCV's equalization routine. The method computes the histogram, cumulative distribution function (CDF), and a lookup table that remaps the image intensities.

Important code used:

```python
hist = np.bincount(img.ravel(), minlength=256)
cdf = hist.cumsum()
nonzero = cdf[cdf > 0]
cdf_min = nonzero[0]
lut = np.round((cdf - cdf_min) * 255.0 / (img.size - cdf_min)).clip(0, 255).astype(np.uint8)
equalized = lut[img]
```

![Q3 comparison](outputs/q3/comparison.png)

The equalized runway image shows improved global contrast. Details in darker and mid-tone regions become easier to distinguish because the distribution of gray levels is spread more widely over the available intensity range.

## Q4. Otsu Thresholding and Foreground Equalization

The woman-in-doorway image was first converted to grayscale, and Otsu thresholding was used to separate the foreground. The threshold obtained was:

- `100.00`

Histogram equalization was then applied only to the foreground mask. This reveals hidden features such as interior room details and subtle textures around the doorway that are less visible in the original grayscale image.

Important code used:

```python
threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
foreground = gray[mask > 0]
hist = np.bincount(foreground, minlength=256)
cdf = hist.cumsum()
lut = np.round((cdf - cdf[cdf > 0][0]) * 255.0 / (foreground.size - cdf[cdf > 0][0])).clip(0, 255).astype(np.uint8)
result[mask > 0] = lut[result[mask > 0]]
```

![Q4 comparison](outputs/q4/comparison.png)

Applying equalization only on the foreground prevents the background from dominating the transformation and gives a more meaningful enhancement of the relevant scene content.

## Q5. Gaussian Filtering

A normalized `5 x 5` Gaussian kernel was computed manually for `sigma = 2`. The resulting kernel was:

```text
[[0.02324684 0.03382395 0.03832756 0.03382395 0.02324684]
 [0.03382395 0.04921356 0.05576627 0.04921356 0.03382395]
 [0.03832756 0.05576627 0.06319146 0.05576627 0.03832756]
 [0.03382395 0.04921356 0.05576627 0.04921356 0.03382395]
 [0.02324684 0.03382395 0.03832756 0.03382395 0.02324684]]
```

The same image was smoothed using both the manually computed kernel and OpenCV's `GaussianBlur()`.

Important code used:

```python
yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
kernel = np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2)))
kernel /= kernel.sum()

for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        region = padded[row:row + kernel.shape[0], col:col + kernel.shape[1]]
        result[row, col] = np.sum(region * kernel)
```

![Q5 comparison](outputs/q5/comparison.png)

Comparison of results:

- The manual Gaussian blur and OpenCV Gaussian blur are visually almost identical.
- Any minor difference comes from padding and implementation details, not from the Gaussian model itself.

The 3D visualization of the larger Gaussian kernel is shown below.

![Q5 kernel surface](outputs/q5/gaussian_kernel_51x51_surface.png)

## Q6. Derivative of Gaussian and Sobel Filtering

The first-order derivatives of the 2D Gaussian were used to construct derivative-of-Gaussian kernels in the `x` and `y` directions for `sigma = 2`. The manually computed `5 x 5` kernels were:

```text
Gx:
[[ 0.04413011  0.03210446 -0.         -0.03210446 -0.04413011]
 [ 0.06420893  0.04671172 -0.         -0.04671172 -0.06420893]
 [ 0.07275825  0.05293131 -0.         -0.05293131 -0.07275825]
 [ 0.06420893  0.04671172 -0.         -0.04671172 -0.06420893]
 [ 0.04413011  0.03210446 -0.         -0.03210446 -0.04413011]]

Gy:
[[ 0.04413011  0.06420893  0.07275825  0.06420893  0.04413011]
 [ 0.03210446  0.04671172  0.05293131  0.04671172  0.03210446]
 [-0.         -0.         -0.         -0.         -0.        ]
 [-0.03210446 -0.04671172 -0.05293131 -0.04671172 -0.03210446]
 [-0.04413011 -0.06420893 -0.07275825 -0.06420893 -0.04413011]]
```

The gradient images computed using derivative-of-Gaussian were compared with OpenCV Sobel results.

Important code used:

```python
gaussian = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2)))
gx = -(xx / sigma**2) * gaussian
gy = -(yy / sigma**2) * gaussian

grad_x = convolve_float(image, gx)
grad_y = convolve_float(image, gy)
magnitude = np.hypot(grad_x, grad_y)
```

![Q6 comparison](outputs/q6/comparison.png)

Comparison of results:

- Derivative-of-Gaussian gives smoother gradients because smoothing and differentiation are combined.
- Sobel gives stronger and sharper responses, but it is generally more sensitive to noise.
- Both highlight the same major boundaries, so the manual implementation is consistent with the OpenCV result.

The 3D surface plot of the derivative-of-Gaussian kernel is shown below.

![Q6 kernel surface](outputs/q6/dog_kernel_x_51x51_surface.png)

## Q7. Image Zooming with Nearest-Neighbor and Bilinear Interpolation

Manual nearest-neighbor and bilinear interpolation functions were implemented and tested using the provided large/small image pairs in `images/q7_images`. The quality was measured using normalized SSD after scaling the small image back to the original size.

Important code used:

```python
top = (1 - dx) * image[y1, x1] + dx * image[y1, x2]
bottom = (1 - dx) * image[y2, x1] + dx * image[y2, x2]
out[row, col] = (1 - dy) * top + dy * bottom

diff = image_a.astype(np.float64) - image_b.astype(np.float64)
ssd = np.mean(diff**2)
```

### SSD Results

| Image Pair | Nearest-Neighbor SSD | Bilinear SSD |
| --- | ---: | ---: |
| `im01 / im01small` | 255.295693 | 200.245140 |
| `im02 / im02small` | 64.629378 | 48.960666 |
| `im03 / im03small` | 147.765454 | 113.566136 |
| `taylor / taylor_small` | 320.088217 | 285.139552 |

Example output:

![Q7 original example](outputs/q7/im02_original.png)
![Q7 bilinear example](outputs/q7/im02_bilinear.png)

Comparison of results:

- In every tested pair, bilinear interpolation produced a lower SSD than nearest-neighbor interpolation.
- Nearest-neighbor produces visible blockiness around edges.
- Bilinear interpolation gives smoother transitions and better approximates the original image.

## Q8. Noise Reduction on Salt-and-Pepper Noise

Gaussian smoothing and median filtering were applied to the noisy image.

Important code used:

```python
gaussian = cv2.GaussianBlur(img, (5, 5), 1.5)
median = cv2.medianBlur(img, 5)
```

![Q8 comparison](outputs/q8/comparison.png)

Comparison of results:

- Gaussian smoothing reduces noise but also blurs edges and spreads impulse artifacts.
- Median filtering removes salt-and-pepper noise more effectively while preserving boundaries better.
- Therefore, median filtering is the better choice for this noise type.

## Q9. Image Sharpening

Image sharpening was performed on `sapphire.jpg` using unsharp masking with:

- `sigma = 1.8`
- `strength = 1.8`

Important code used:

```python
blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
sharp = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
```

![Q9 comparison](outputs/q9/comparison.png)

Comparison of results:

- The sharpened image shows stronger edges and clearer gem facets than the original.
- Fine texture is enhanced, but slight noise amplification is also visible in detailed areas.

## Q10. Bilateral Filtering

A bilateral filter was implemented manually for grayscale images and tested on `einstein.png` after converting it to grayscale. The result was compared against Gaussian smoothing and OpenCV's bilateral filter.

Parameters used:

- Manual bilateral: `diameter = 7`, `sigma_s = 7.0`, `sigma_r = 35.0`
- OpenCV bilateral: `d = 7`, `sigmaColor = 35`, `sigmaSpace = 7`

Important code used:

```python
spatial = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_s**2))
range_weights = np.exp(-((region - center) ** 2) / (2.0 * sigma_r**2))
weights = spatial * range_weights
weights /= weights.sum()
out[row, col] = np.sum(weights * region)
```

![Q10 comparison](outputs/q10/comparison.png)

Comparison of results:

- Gaussian blur smooths both noise and edges.
- Bilateral filtering smooths noise while preserving edge transitions better.
- On the Einstein image, facial boundaries and hair contours remain clearer after bilateral filtering, while Gaussian smoothing produces a softer appearance.
- The manual bilateral output and OpenCV bilateral output are similar, which validates the manual implementation.

## Q11. Relationship Between Spatial Filtering and Frequency Response

By the convolution theorem, spatial-domain convolution corresponds to multiplication in the frequency domain:

`g(x, y) = f(x, y) * h(x, y)`  corresponds to  `G(u, v) = F(u, v)H(u, v)`

This means that filtering can be interpreted as selectively attenuating or amplifying certain frequency components.

- Averaging filter: acts as a low-pass filter, suppressing high-frequency detail but possibly causing ripple effects.
- Gaussian filter: also low-pass, but with a smoother frequency response and fewer artifacts.
- Laplacian filter: acts as a high-pass filter, emphasizing edges and fine detail.

Gaussian filtering avoids ringing because its frequency response changes smoothly, unlike the abrupt cutoff of an ideal low-pass filter. For high-frequency noise reduction, the Gaussian filter is the most suitable among the listed filters because it suppresses noise smoothly in both the spatial and frequency domains.

## Q12. Homomorphic Filtering for Illumination Correction

Homomorphic filtering models an image as the product of illumination and reflectance:

`f(x, y) = i(x, y) * r(x, y)`

Taking the logarithm converts this multiplicative model into an additive form, which allows low-frequency illumination and high-frequency reflectance to be manipulated separately in the frequency domain.

The implemented approach was:

1. Convert the image to a luminance channel.
2. Apply logarithmic transformation.
3. Compute the Fourier transform.
4. Apply a high-frequency emphasis filter.
5. Compute the inverse Fourier transform.
6. Apply the inverse log operation and normalize.

Important code used:

```python
log_image = np.log(gray.astype(np.float32) + 1.0)
shifted = np.fft.fftshift(np.fft.fft2(log_image))
filtered = filter_response * shifted
restored = np.fft.ifft2(np.fft.ifftshift(filtered))
exp_image = np.exp(np.real(restored)) - 1.0
```

![Q12 comparison](outputs/q12/comparison.png)

Comparison of results:

- Histogram equalization changes contrast globally without modeling illumination.
- Homomorphic filtering is better when the image suffers from non-uniform lighting or shadows.
- The result here improves illumination balance and local contrast, with a small risk of halo artifacts if the filter is too aggressive.

## Conclusion

This assignment demonstrated how point operations and neighborhood filtering can be used for image enhancement, edge extraction, denoising, resizing, and illumination correction. Manual implementations of histogram equalization, Gaussian filtering, derivative-of-Gaussian filtering, bilateral filtering, and interpolation produced results consistent with standard OpenCV functions. Overall, the experiments showed the trade-offs between enhancement strength, edge preservation, noise suppression, and computational complexity.
