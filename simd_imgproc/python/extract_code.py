#!/usr/bin/env python3
"""Cuts the code excerpts shown in docs/guide.tex out of the real source files, so the listings in the PDF cannot
drift away from the code that is actually built and tested.
Excerpt = lines from the first line containing START up to (not including) the first later line containing END
(END = None: to the end of the file)."""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "snippets")

EXCERPTS = [
    # name,                file,                         start marker,                                  end marker
    ("image_class",        "include/simdip/image.hpp",   "class Image {",                               "// ---- file I/O"),
    ("scalar_helpers",     "src/scalar_impl.inc",        "inline int clampInt",                         "void rgbToGrayRows"),
    ("scalar_gray_sobel",  "src/scalar_impl.inc",        "void rgbToGrayRows",                          "void convolveRows"),
    ("scalar_conv",        "src/scalar_impl.inc",        "void convolveRows",                           None),
    ("omp_bands",          "src/dispatch.cpp",           "// Calls f(y0, y1)",                          "void checkPair"),
    ("avx2_masks",         "src/avx2.cpp",               "struct DeinterleaveMasks {",                  "void rgbToGrayRows"),
    ("avx2_gray",          "src/avx2.cpp",               "void rgbToGrayRows",                          "// 2. Sobel"),
    ("avx2_sobel",         "src/avx2.cpp",               "void sobelRows",                              "// 3. Convolution"),
    ("avx2_pack",          "src/avx2.cpp",               "// Turns 4 vectors of 8 int32",               "struct FreeDeleter"),
    ("avx2_convert_row",   "src/avx2.cpp",               "// Replicated-border float copy",             "// NV = number of 8-float"),
    ("avx2_conv",          "src/avx2.cpp",               "// NV = number of 8-float",                   "}  // namespace"),
]


def extract(path, start, end):
    lines = open(os.path.join(ROOT, path)).read().split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next((k for k in range(i + 1, len(lines)) if end in lines[k]), len(lines)) if end else len(lines)
    chunk = lines[i:j]
    while chunk and (not chunk[-1].strip() or chunk[-1].strip().startswith(("// ----", "// ===", "}  // namespace"))):
        chunk.pop()
    return "\n".join(chunk) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, path, start, end in EXCERPTS:
        text = extract(path, start, end)
        with open(os.path.join(OUT, f"code_{name}.cpp"), "w") as f:
            f.write(text)
        print(f"code_{name}.cpp: {text.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
