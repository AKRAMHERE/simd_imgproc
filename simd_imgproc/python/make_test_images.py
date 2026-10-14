#!/usr/bin/env python3
"""Writes the test photographs and a few synthetic images to data/ (PGM gray, PPM colour)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pnm import write_pnm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")


def main():
    os.makedirs(OUT, exist_ok=True)
    from skimage import data

    cam = data.camera()                      # 512x512 gray
    ast = data.astronaut()                   # 512x512 RGB
    chel = data.chelsea()                    # 300x451 RGB (odd size on purpose)
    write_pnm(os.path.join(OUT, "camera.pgm"), cam)
    write_pnm(os.path.join(OUT, "astronaut.ppm"), ast)
    write_pnm(os.path.join(OUT, "chelsea.ppm"), chel)
    write_pnm(os.path.join(OUT, "chelsea_gray.pgm"), (chel @ np.array([0.299, 0.587, 0.114])).round().astype(np.uint8))
    rng = np.random.default_rng(1)
    write_pnm(os.path.join(OUT, "noise.pgm"), rng.integers(0, 256, (333, 517), dtype=np.uint8))
    write_pnm(os.path.join(OUT, "noise_rgb.ppm"), rng.integers(0, 256, (101, 259, 3), dtype=np.uint8))
    # checkerboard of 0/255: strongest possible edges everywhere
    yy, xx = np.mgrid[0:128, 0:200]
    write_pnm(os.path.join(OUT, "checker.pgm"), (((xx + yy) & 1) * 255).astype(np.uint8))
    print("wrote test images to", OUT)


if __name__ == "__main__":
    main()
