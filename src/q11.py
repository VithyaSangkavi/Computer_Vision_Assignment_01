from utils import output_path, write_text


ANSWERS = """Q11. Relationship Between Spatial Filtering and Frequency Response

(a) By the convolution theorem, spatial-domain convolution g(x, y) = f(x, y) * h(x, y)
corresponds to multiplication in the frequency domain:
G(u, v) = F(u, v)H(u, v).
This means the filter spectrum H(u, v) scales each frequency component of the image.
Low frequencies usually represent slowly varying illumination and coarse structure, while
high frequencies represent fine detail, edges, and noise.

(b) Qualitative frequency-domain effects
- Averaging (box) filter: behaves like a low-pass filter. It attenuates high frequencies, so
  the image becomes smoother, but its sinc-like response can introduce ripples and ringing.
- Gaussian filter: also low-pass, but with a smooth Gaussian response and no abrupt cutoff.
  It suppresses high frequencies smoothly, producing natural blur with fewer artifacts.
- Laplacian filter: behaves like a high-pass operator because it suppresses slowly varying
  regions and emphasizes rapid intensity changes. Spatially, this highlights edges and fine detail.

(c) Gaussian filtering avoids ringing because its frequency response changes smoothly rather than
having the sharp cutoff of an ideal low-pass filter. Sharp cutoffs create oscillations in the
spatial domain, which appear as ringing near edges.

(d) For high-frequency noise reduction, the Gaussian filter is the most suitable among these options.
In the spatial domain it performs localized weighted averaging, giving more weight to nearby pixels.
In the frequency domain it suppresses high frequencies smoothly, reducing noise without the stronger
ripples of a box filter. The Laplacian is unsuitable because it amplifies high-frequency content,
including noise.
"""


def run() -> None:
    write_text(output_path("q11", "answers.txt"), ANSWERS)


if __name__ == "__main__":
    run()
