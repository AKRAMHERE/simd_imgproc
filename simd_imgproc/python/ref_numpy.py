"""Independent NumPy / SciPy implementations of the three operations (no code shared with the C++ library).

gray   : integer fixed point  Y = (77 R + 150 G + 29 B + 128) >> 8
sobel  : min(255, |Gx| + |Gy|) with replicated border, int32 arithmetic
conv   : scipy.ndimage.correlate in float64, mode='nearest' (= replicate border), np.rint (ties to even), clip
"""
import numpy as np
from scipy import ndimage


def gray(rgb):
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return ((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8)


def sobel(img):
    p = np.pad(img.astype(np.int32), 1, mode="edge")
    a, b, c = p[:-2], p[1:-1], p[2:]                       # rows y-1, y, y+1 (full padded width)
    gx = (a[:, 2:] - a[:, :-2]) + 2 * (b[:, 2:] - b[:, :-2]) + (c[:, 2:] - c[:, :-2])
    gy = (c[:, :-2] + 2 * c[:, 1:-1] + c[:, 2:]) - (a[:, :-2] + 2 * a[:, 1:-1] + a[:, 2:])
    return np.minimum(np.abs(gx) + np.abs(gy), 255).astype(np.uint8)


def conv(img, kernel):
    out = ndimage.correlate(img.astype(np.float64), np.asarray(kernel, dtype=np.float64), mode="nearest")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---- kernels that numpy can build bit-exactly like the C++ code (float32 weights) ----
def k_identity(n):
    k = np.zeros((n, n), np.float32)
    k[n // 2, n // 2] = 1
    return k


def k_box(n):
    return np.full((n, n), np.float32(1) / np.float32(n * n), np.float32)


def k_binomial(n):
    row = np.array([1.0])
    for _ in range(n - 1):
        row = np.convolve(row, [1.0, 1.0])
    return (np.outer(row, row) / (2.0 ** (2 * (n - 1)))).astype(np.float32)


def k_sharpen():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)


def k_emboss():
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)


def k_gauss(n, sigma):
    r = n // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return (g / g.sum()).astype(np.float32)
