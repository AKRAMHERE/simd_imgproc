#!/usr/bin/env python3
"""Runs the C++ command line tool on many images / implementations / thread counts and compares the output with
the independent NumPy+SciPy implementation (python/ref_numpy.py).

Expected:  gray and sobel   -> identical (0 differing pixels)
           conv, exact kernels (identity, binomial, sharpen, emboss)  -> identical
           conv, other kernels (box, gauss): float32 vs float64 rounding may flip a value that sits within ~1e-6
           of a tie (x.5) by one grey level; we require max difference <= 1 and report the fraction.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ref_numpy as ref  # noqa: E402
from pnm import read_pnm, write_pnm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "build", "simdip")


def cpu_has_avx2():
    out = subprocess.run([CLI, "info"], capture_output=True, text=True).stdout
    return "avx2+fma:   yes" in out


def run(args):
    r = subprocess.run([CLI] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(args) + ": " + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    impls = ["scalar", "autovec"] + (["avx2"] if cpu_has_avx2() else [])
    threads = [1, 3]
    rng = np.random.default_rng(7)
    data = os.path.join(ROOT, "data")

    rgb_files = ["astronaut.ppm", "chelsea.ppm", "noise_rgb.ppm"]
    gray_files = ["camera.pgm", "chelsea_gray.pgm", "noise.pgm", "checker.pgm"]
    rgb_imgs = {f: read_pnm(os.path.join(data, f)) for f in rgb_files}
    gray_imgs = {f: read_pnm(os.path.join(data, f)) for f in gray_files}
    # random shapes around the vector widths (also 1-pixel wide/high images)
    for (h, w) in [(1, 1), (1, 40), (40, 1), (5, 17), (9, 33), (3, 100), (64, 64), (37, 53), (130, 31)]:
        rgb_imgs[f"rand_{w}x{h}.ppm"] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        gray_imgs[f"rand_{w}x{h}.pgm"] = rng.integers(0, 256, (h, w), dtype=np.uint8)
    if args.quick:
        rgb_imgs = dict(list(rgb_imgs.items())[:4])
        gray_imgs = dict(list(gray_imgs.items())[:5])

    exact_kernels = [("identity", 3, ref.k_identity(3)), ("identity", 7, ref.k_identity(7)),
                     ("binomial", 3, ref.k_binomial(3)), ("binomial", 5, ref.k_binomial(5)),
                     ("binomial", 9, ref.k_binomial(9)), ("sharpen", 3, ref.k_sharpen()), ("emboss", 3, ref.k_emboss())]
    approx_kernels = [("box", 3, ref.k_box(3)), ("box", 5, ref.k_box(5)), ("box", 11, ref.k_box(11)),
                      ("gauss", 5, ref.k_gauss(5, 1.0)), ("gauss", 9, ref.k_gauss(9, 2.0))]

    comparisons = mismatches = 0
    approx_pixels = approx_diff = 0
    worst = 0
    tmp = tempfile.mkdtemp()
    inp, outp = os.path.join(tmp, "in"), os.path.join(tmp, "out.pgm")

    def compare(tag, got, want, exact=True):
        nonlocal comparisons, mismatches, approx_pixels, approx_diff, worst
        comparisons += 1
        d = np.abs(got.astype(int) - want.astype(int))
        if exact:
            if d.max() != 0:
                mismatches += 1
                print("MISMATCH", tag, "max diff", d.max())
        else:
            approx_pixels += d.size
            approx_diff += int((d != 0).sum())
            worst = max(worst, int(d.max()))
            if d.max() > 1:
                mismatches += 1
                print("MISMATCH (>1)", tag, "max diff", d.max())

    for impl in impls:
        for t in threads:
            common = ["--impl", impl, "--threads", str(t)]
            for name, img in rgb_imgs.items():
                write_pnm(inp + ".ppm", img)
                run(["gray", inp + ".ppm", outp] + common)
                compare(f"gray {name} {impl} t={t}", read_pnm(outp), ref.gray(img))
            for name, img in gray_imgs.items():
                write_pnm(inp + ".pgm", img)
                run(["sobel", inp + ".pgm", outp] + common)
                compare(f"sobel {name} {impl} t={t}", read_pnm(outp), ref.sobel(img))
                for kname, size, kern in exact_kernels:
                    run(["conv", inp + ".pgm", outp, "--kernel", kname, "--size", str(size)] + common)
                    compare(f"conv {kname}{size} {name} {impl} t={t}", read_pnm(outp), ref.conv(img, kern))
                if name.startswith(("camera", "chelsea", "rand_37", "rand_64")):
                    for kname, size, kern in approx_kernels:
                        extra = ["--sigma", "1.0" if size == 5 else "2.0"] if kname == "gauss" else []
                        run(["conv", inp + ".pgm", outp, "--kernel", kname, "--size", str(size)] + extra + common)
                        compare(f"conv {kname}{size} {name} {impl} t={t}", read_pnm(outp), ref.conv(img, kern), exact=False)

    print(f"{comparisons} comparisons, {mismatches} mismatches")
    if approx_pixels:
        print(f"non-exact kernels (box, gauss): {approx_diff} of {approx_pixels} pixels differ by one grey level "
              f"({100.0 * approx_diff / approx_pixels:.5f} %), worst difference {worst}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
