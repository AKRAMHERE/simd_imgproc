#!/usr/bin/env python3
"""Sanity check and speed comparison against OpenCV (cv2), single-threaded.

  1. agreement: do OpenCV's cvtColor / Sobel / filter2D give (almost) the same pixels as this library?
  2. speed: median time of OpenCV vs the C++ benchmark's AVX2 single-thread numbers (read from a bench CSV).

CAVEATS written into the output on purpose: OpenCV's colour conversion uses 14-bit coefficients (so it differs from our
8-bit coefficients by up to 1 level); its Sobel is a 3-call pipeline here (two Sobel calls + abs/saturate + add) while ours is
one fused pass; OpenCV also allocates its output on every call. The comparison shows the order of magnitude,
not which library is "better".
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ref_numpy as ref  # noqa: E402
from pnm import read_pnm, write_pnm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "build", "simdip")


def median_ms(fn, reps=9, min_ms=30):
    fn()
    fn()
    t0 = time.perf_counter()
    fn()
    one = (time.perf_counter() - t0) * 1e3
    iters = max(1, int(min_ms / max(one, 1e-6)) + 1)
    s = []
    for _ in range(reps):
        a = time.perf_counter()
        for _ in range(iters):
            fn()
        s.append((time.perf_counter() - a) * 1e3 / iters)
    return statistics.median(s)


def cv_sobel(img):
    gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return cv2.add(cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-csv", default=None, help="CSV from simdip_bench (avx2, 1 thread) to compare speed with")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "opencv_compare.csv"))
    args = ap.parse_args()
    cv2.setNumThreads(1)
    print("OpenCV", cv2.__version__, " threads:", cv2.getNumThreads(), " optimized:", cv2.useOptimized())

    # ---------------- 1. agreement on real photographs
    ast = read_pnm(os.path.join(ROOT, "data", "astronaut.ppm"))
    cam = read_pnm(os.path.join(ROOT, "data", "camera.pgm"))
    d = np.abs(cv2.cvtColor(ast, cv2.COLOR_RGB2GRAY).astype(int) - ref.gray(ast).astype(int))
    print(f"gray   vs cv2.cvtColor : max |diff| = {d.max()}, differing pixels = {100.0 * (d != 0).mean():.2f} %")
    d = np.abs(cv_sobel(cam).astype(int) - ref.sobel(cam).astype(int))
    print(f"sobel  vs cv2.Sobel x2 : max |diff| = {d.max()}, differing pixels = {100.0 * (d != 0).mean():.4f} %")
    for k in (3, 5, 9):
        kern = ref.k_gauss(k, max(0.8, k / 3.0))
        d = np.abs(cv2.filter2D(cam, -1, kern, borderType=cv2.BORDER_REPLICATE).astype(int) - ref.conv(cam, kern).astype(int))
        print(f"conv{k}  vs cv2.filter2D : max |diff| = {d.max()}, differing pixels = {100.0 * (d != 0).mean():.4f} %")

    # ---------------- 2. speed
    n = args.size
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    g = rng.integers(0, 256, (n, n), dtype=np.uint8)
    rows = []
    rows.append(("gray", 0, "opencv", median_ms(lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))))
    rows.append(("sobel", 0, "opencv", median_ms(lambda: cv_sobel(g))))
    for k in (3, 5, 9):
        kern = ref.k_gauss(k, max(0.8, k / 3.0))
        rows.append(("conv", k, "opencv", median_ms(lambda kern=kern: cv2.filter2D(g, -1, kern, borderType=cv2.BORDER_REPLICATE))))
    mine = {}
    if args.bench_csv:
        with open(args.bench_csv) as f:
            for r in csv.DictReader(f):
                if r["impl"] == "avx2" and int(r["threads"]) == 1 and int(r["width"]) == n:
                    mine[(r["op"], int(r["ksize"]))] = float(r["median_ms"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "ksize", "size", "opencv_ms", "simdip_avx2_ms", "opencv_over_simdip"])
        print(f"\n{'op':6s} {'k':>3s} {'OpenCV ms':>10s} {'simdip avx2 ms':>15s} {'ratio':>7s}")
        for op, k, _, t in rows:
            m = mine.get((op, k))
            ratio = t / m if m else float("nan")
            w.writerow([op, k, n, f"{t:.4f}", f"{m:.4f}" if m else "", f"{ratio:.2f}" if m else ""])
            print(f"{op:6s} {k:3d} {t:10.3f} {m if m else float('nan'):15.3f} {ratio:7.2f}")


if __name__ == "__main__":
    main()
