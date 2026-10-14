#!/usr/bin/env python3
"""Generates every figure, table and number used by docs/guide.tex from the code and from the measured CSV files.

  python3 python/make_figures.py [--results results/sandbox_example] [--out DIR] [--testlog F] [--crosslog F]

Outputs (default): docs/figures/*.pdf, docs/snippets/*.tex (tables), docs/snippets/macros.tex (numbers as LaTeX macros).
With --out DIR the figures go to DIR/figures and the tables/macros to DIR/snippets, and docs/ is not touched; use this for
cluster results:  python3 python/make_figures.py --results results/cpu_il --out results/cpu_il/report
(needs the files written by slurm/02_bench_single.sbatch; an E7_scaling.csv from slurm/03 adds a scaling figure and table).
Nothing in the guide that is a measurement or a computed value is typed by hand: it comes from here.
"""
import argparse
import collections
import csv
import os
import re
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import ref_numpy as ref  # noqa: E402
from pnm import read_pnm, write_pnm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
SNIP = os.path.join(ROOT, "docs", "snippets")
CLI = os.path.join(ROOT, "build", "simdip")

C = {"scalar": "#7f7f7f", "autovec": "#d98c1f", "avx2": "#1f5fa8", "avx2x2": "#2e7d32", "other": "#8e44ad"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25})

MACROS = collections.OrderedDict()


def macro(name, value):
    MACROS[name] = str(value)


def write(name, text):
    """Writes a snippet. A table body must not end in a row terminator: the chapters append `\\\\ \\bottomrule`
    themselves (see \\inputrows in guide.tex)."""
    text = text.rstrip()
    if text.endswith("\\\\"):
        text = text[:-2].rstrip()
    with open(os.path.join(SNIP, name), "w") as f:
        f.write(text)


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


class Data:
    def __init__(self, rows):
        self.rows = rows

    def ms(self, op, impl, thr, size, k=0):
        for r in self.rows:
            if (r["op"], r["impl"], int(r["threads"]), int(r["width"]), int(r["ksize"])) == (op, impl, thr, size, k):
                return float(r["median_ms"])
        raise KeyError((op, impl, thr, size, k))

    def get(self, op, impl, thr, size, k=0, field="median_ms"):
        for r in self.rows:
            if (r["op"], r["impl"], int(r["threads"]), int(r["width"]), int(r["ksize"])) == (op, impl, thr, size, k):
                return float(r[field])
        raise KeyError((op, impl, thr, size, k))


def run_cli(args):
    r = subprocess.run([CLI] + args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)


def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ------------------------------------------------------------------------------------------------ figures: gallery
def fig_gallery():
    cam = read_pnm(os.path.join(ROOT, "data", "camera.pgm"))
    ast = read_pnm(os.path.join(ROOT, "data", "astronaut.ppm"))
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "o.pgm")
    cam_p, ast_p = os.path.join(tmp, "cam.pgm"), os.path.join(tmp, "ast.ppm")
    write_pnm(cam_p, cam)
    write_pnm(ast_p, ast)
    res = {}
    for name, args in [("Gaussian 9x9, sigma 2", ["conv", cam_p, out, "--kernel", "gauss", "--size", "9", "--sigma", "2"]),
                       ("sharpen 3x3", ["conv", cam_p, out, "--kernel", "sharpen"]),
                       ("emboss 3x3", ["conv", cam_p, out, "--kernel", "emboss"]),
                       ("box 15x15", ["conv", cam_p, out, "--kernel", "box", "--size", "15"]),
                       ("Sobel", ["sobel", cam_p, out])]:
        run_cli(args)
        res[name] = read_pnm(out)
    run_cli(["gray", ast_p, out])
    gray = read_pnm(out)
    tiles = [("input (gray)", cam, "gray"), ("Gaussian 9x9, sigma 2", res["Gaussian 9x9, sigma 2"], "gray"),
             ("sharpen 3x3", res["sharpen 3x3"], "gray"), ("Sobel", res["Sobel"], "gray"),
             ("input (RGB)", ast, None), ("RGB to gray", gray, "gray"), ("emboss 3x3", res["emboss 3x3"], "gray"),
             ("box 15x15", res["box 15x15"], "gray")]
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.9))
    for ax, (t, im, cm) in zip(axes.ravel(), tiles):
        ax.imshow(im, cmap=cm, vmin=0, vmax=255)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_ops_gallery.pdf"))
    plt.close(fig)


def fig_sobel_parts():
    cam = read_pnm(os.path.join(ROOT, "data", "camera.pgm")).astype(np.int32)
    p = np.pad(cam, 1, mode="edge")
    a, b, c = p[:-2], p[1:-1], p[2:]
    gx = (a[:, 2:] - a[:, :-2]) + 2 * (b[:, 2:] - b[:, :-2]) + (c[:, 2:] - c[:, :-2])
    gy = (c[:, :-2] + 2 * c[:, 1:-1] + c[:, 2:]) - (a[:, :-2] + 2 * a[:, 1:-1] + a[:, 2:])
    mag = ref.sobel(cam.astype(np.uint8))
    assert (np.minimum(np.abs(gx) + np.abs(gy), 255) == mag).all()
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6))
    for ax, (t, im) in zip(axes, [("input", cam), (r"$|G_x|$ (clipped to 255)", np.minimum(np.abs(gx), 255)),
                                  (r"$|G_y|$ (clipped to 255)", np.minimum(np.abs(gy), 255)), (r"$\min(255,|G_x|+|G_y|)$", mag)]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_sobel_parts.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ results
def label(op, k):
    return {"gray": "gray", "sobel": "Sobel"}.get(op, f"conv {k}x{k}")


def results(res):
    meta = {}
    for line in open(os.path.join(res, "meta.txt")):
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    macro("sbxCpu", tex_escape(meta.get("cpu", "?")))
    macro("sbxCompiler", tex_escape(meta.get("compiler", "?")))
    macro("sbxDate", tex_escape(meta.get("date", "?").split()[0]))
    E1 = Data(load(os.path.join(res, "E1_main.csv")))
    E2 = Data(load(os.path.join(res, "E2_sizes.csv")))
    E4 = Data(load(os.path.join(res, "E4_ksize.csv")))
    E5 = Data(load(os.path.join(res, "E5_copy.csv")))

    # ---- between-pass variation: the first pass of E1 (kept in firstrun/) versus the repeat of the same command
    first = os.path.join(res, "firstrun", "E1_main.csv")
    if os.path.exists(first):
        A = {(r["op"], r["impl"], r["threads"], r["width"], r["ksize"]): float(r["median_ms"]) for r in load(first)}
        Bm = {(r["op"], r["impl"], r["threads"], r["width"], r["ksize"]): float(r["median_ms"]) for r in E1.rows}
        by = collections.defaultdict(list)
        for k, a in A.items():
            if k in Bm and int(k[3]) >= 1024:
                by[k[1]].append((max(a, Bm[k]) / min(a, Bm[k]) - 1) * 100)
        for impl, name in (("scalar", "Scalar"), ("autovec", "Autovec"), ("avx2", "Avx")):
            v = sorted(by.get(impl, [0]))
            macro(f"between{name}Med", fmt(v[len(v) // 2], 0))
            macro(f"between{name}Max", fmt(v[-1], 0))

    # ---- repeatability of the sandbox
    spread1 = spread2 = 0.0
    groups = collections.defaultdict(list)
    for i in (1, 2, 3):
        for r in load(os.path.join(res, f"E6_repeat_{i}.csv")):
            groups[(r["op"], r["impl"], r["threads"], r["ksize"])].append(float(r["median_ms"]))
    for key, v in groups.items():
        s = (max(v) - min(v)) / min(v) * 100
        if key[2] == "1":
            spread1 = max(spread1, s)
        else:
            spread2 = max(spread2, s)
    macro("repSpreadOne", f"{spread1:.0f}")
    macro("repSpreadTwo", f"{spread2:.0f}")

    # ---- E1 table at 2048 and speedup figure
    SIZE = 2048
    cases = [("gray", 0), ("sobel", 0), ("conv", 3), ("conv", 5), ("conv", 9), ("conv", 11)]
    rows = []
    sp = {}
    for op, k in cases:
        base = E1.ms(op, "scalar", 1, SIZE, k)
        vals = {"scalar": base, "autovec": E1.ms(op, "autovec", 1, SIZE, k), "avx2": E1.ms(op, "avx2", 1, SIZE, k),
                "scalar2": E1.ms(op, "scalar", 2, SIZE, k), "avx22": E1.ms(op, "avx2", 2, SIZE, k)}
        sp[(op, k)] = {n: base / v for n, v in vals.items()}
        rows.append(f"{label(op, k)} & {vals['scalar']:.2f} & {vals['autovec']:.2f} ({sp[(op, k)]['autovec']:.1f}$\\times$) & "
                    f"{vals['avx2']:.2f} ({sp[(op, k)]['avx2']:.1f}$\\times$) & {vals['scalar2']:.2f} ({sp[(op, k)]['scalar2']:.1f}$\\times$) & "
                    f"{vals['avx22']:.2f} ({sp[(op, k)]['avx22']:.1f}$\\times$) \\\\")
        key = {("gray", 0): "Gray", ("sobel", 0): "Sobel", ("conv", 3): "ConvThree", ("conv", 5): "ConvFive",
               ("conv", 9): "ConvNine", ("conv", 11): "ConvEleven"}[(op, k)]
        macro(f"ms{key}Scalar", fmt(vals["scalar"], 2))
        macro(f"sp{key}Autovec", fmt(sp[(op, k)]["autovec"], 1))
        macro(f"sp{key}Avx", fmt(sp[(op, k)]["avx2"], 1))
        macro(f"sp{key}AvxTwo", fmt(sp[(op, k)]["avx22"], 1))
        macro(f"sp{key}ScalarTwo", fmt(sp[(op, k)]["scalar2"], 1))
        macro(f"sc{key}Avx", fmt(vals["avx2"] / vals["avx22"], 2))   # threads scaling of avx2: t1 / t2
    write("tab_e1.tex", "\n".join(rows))

    # figure: speedups
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    names = [label(o, k) for o, k in cases]
    x = np.arange(len(cases))
    w = 0.16
    series = [("autovec", "autovec", C["autovec"]), ("avx2", "AVX2", C["avx2"]), ("scalar2", "scalar, 2 threads", C["scalar"]),
              ("avx22", "AVX2, 2 threads", C["avx2x2"])]
    for i, (key, lab, col) in enumerate(series):
        v = [sp[c][key] for c in cases]
        bars = ax.bar(x + (i - 1.5) * w, v, w, label=lab, color=col)
        for b_, vv in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.4, f"{vv:.0f}" if vv >= 9.5 else f"{vv:.1f}", ha="center", fontsize=6.5)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("speedup over scalar, 1 thread")
    ax.set_title(f"Speedup at {SIZE}x{SIZE} pixels (sandbox VM, 2 cores; median of 7)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_speedup.pdf"))
    plt.close(fig)

    # ---- E4: convolution versus kernel size
    ks = sorted({int(r["ksize"]) for r in E4.rows})
    impls = []
    for r in E4.rows:
        if r["impl"] not in impls:
            impls.append(r["impl"])
    nice = {"scalar": "scalar", "autovec": "autovec (natural loops)", "avx2": "AVX2 (8 accumulators)", "avx2_acc4": "AVX2, 4 accumulators",
            "avx2_pertap": "AVX2, convert per tap", "autovec_rowacc": "autovec, row-accumulate rewrite"}
    col = {"scalar": C["scalar"], "autovec": C["autovec"], "avx2": C["avx2"], "avx2_acc4": "#5aa0e0", "avx2_pertap": "#9b59b6",
           "autovec_rowacc": "#e0a030"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i in impls:
        ax.plot(ks, [E4.get("conv", i, 1, 1024, k, "gflops") for k in ks], "o-", ms=3.5, label=nice[i], color=col[i])
    ax.set_xlabel("kernel size k (k x k taps per pixel)")
    ax.set_ylabel("GFLOP/s  (2 k^2 flops per pixel)")
    ax.set_title("Convolution throughput versus kernel size, 1024x1024, 1 thread")
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_conv_ksize.pdf"))
    plt.close(fig)
    rows = []
    for i in impls:
        rows.append(f"{nice[i]} & " + " & ".join(f"{E4.get('conv', i, 1, 1024, k, 'gflops'):.1f}" for k in (3, 9, 31)) + " \\\\")
    write("tab_ablation.tex", "\n".join(rows))
    g = lambda i, k: E4.get("conv", i, 1, 1024, k, "gflops")
    macro("gfAvxThree", fmt(g("avx2", 3), 0))
    macro("gfAvxNine", fmt(g("avx2", 9), 0))
    macro("gfAvxMax", fmt(max(g("avx2", k) for k in ks), 0))
    macro("gfScalarMax", fmt(max(g("scalar", k) for k in ks), 1))
    macro("gfRowAccNine", fmt(g("autovec_rowacc", 9), 1))
    macro("gfRowAccMax", fmt(max(g("autovec_rowacc", k) for k in ks), 0))
    macro("ratioAccFourNine", fmt(g("avx2", 9) / g("avx2_acc4", 9), 2))
    macro("ratioAccFourMax", fmt(max(g("avx2", k) / g("avx2_acc4", k) for k in ks), 2))
    macro("ratioPerTapNine", fmt(g("avx2", 9) / g("avx2_pertap", 9), 2))
    macro("ratioRowAccToAvx", fmt(g("avx2", 9) / g("autovec_rowacc", 9), 1))
    macro("ratioRowAccToNatural", fmt(g("autovec_rowacc", 9) / g("autovec", 9), 1))
    # nominal peak: 2 FMA units x 8 lanes x 2 flop x 2.1 GHz
    macro("nominalPeakGf", "67")
    macro("fracOfNominalPeak", fmt(100.0 * max(g("avx2", k) for k in ks) / 67.2, 0))

    # ---- E2: size sweep
    sizes = sorted({int(r["width"]) for r in E2.rows})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for ax, (op, k, title) in zip(axes, [("gray", 0, "gray"), ("sobel", 0, "Sobel"), ("conv", 5, "convolution 5x5")]):
        for impl, colr in (("scalar", C["scalar"]), ("avx2", C["avx2"])):
            ax.plot(sizes, [E2.get(op, impl, 1, s, k, "mpix_s") for s in sizes], "o-", ms=3, color=colr, label=impl)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("image side (pixels)")
        ax.set_title(title, fontsize=9)
        l2_bytes = 2 * 1024 * 1024
        per_px = 4 if op == "gray" else 2
        side = (l2_bytes / per_px) ** 0.5
        ax.axvline(side, color="r", ls="--", lw=0.8)
        ax.text(side * 1.05, ax.get_ylim()[0] * 1.5, "src+dst = 2 MiB (L2)", rotation=90, fontsize=6.5, color="r", va="bottom")
    axes[0].set_ylabel("Mpixel/s (output pixels)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Throughput versus image size, 1 thread: the cache hierarchy shows up", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_sizes.pdf"))
    plt.close(fig)
    small = {op: E2.get(op, "avx2", 1, 512, k, "mpix_s") for op, k in (("gray", 0), ("sobel", 0))}
    big = {op: E2.get(op, "avx2", 1, 8192, k, "mpix_s") for op, k in (("gray", 0), ("sobel", 0))}
    macro("mpixGrayAvxSmall", fmt(small["gray"], 0))
    macro("mpixGrayAvxHuge", fmt(big["gray"], 0))
    macro("mpixSobelAvxSmall", fmt(small["sobel"], 0))
    macro("mpixSobelAvxHuge", fmt(big["sobel"], 0))
    macro("mpixGrayScalarMin", fmt(min(E2.get("gray", "scalar", 1, s, 0, "mpix_s") for s in sizes), 0))
    macro("mpixGrayScalarMax", fmt(max(E2.get("gray", "scalar", 1, s, 0, "mpix_s") for s in sizes), 0))

    # ---- E5: bandwidth yardstick versus the memory-bound kernels
    cp = {(int(r["threads"]), int(r["width"])): float(r["gb_s"]) for r in E5.rows}
    macro("copyOneBig", fmt(cp[(1, 4096)], 0))
    macro("copyTwoBig", fmt(cp[(2, 4096)], 0))
    macro("copyOneSmall", fmt(cp[(1, 512)], 0))
    rows = []
    bars = []
    for size in (2048, 4096):
        for thr in (1, 2):
            gray = E1.get("gray", "avx2", thr, size, 0, "gb_s")
            sob = E1.get("sobel", "avx2", thr, size, 0, "gb_s")
            c = cp[(thr, size)]
            rows.append(f"{size}$^2$ & {thr} & {c:.1f} & {gray:.1f} ({100 * gray / c:.0f}\\,\\%) & {sob:.1f} ({100 * sob / c:.0f}\\,\\%) \\\\")
            bars.append((size, thr, c, gray, sob))
    write("tab_bandwidth.tex", "\n".join(rows))
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    xl = [f"{s}^2, {t} thr" for s, t, *_ in bars]
    x = np.arange(len(bars))
    ax.bar(x - 0.27, [b[2] for b in bars], 0.27, label="memcpy (copy yardstick)", color="#555555")
    ax.bar(x, [b[3] for b in bars], 0.27, label="gray AVX2", color=C["avx2"])
    ax.bar(x + 0.27, [b[4] for b in bars], 0.27, label="Sobel AVX2", color=C["avx2x2"])
    ax.set_xticks(x)
    ax.set_xticklabels(xl, fontsize=7.5)
    ax.set_ylabel("GB/s (minimum traffic: bytes read + written)")
    ax.set_title("How close do the memory-bound kernels get to the copy bandwidth?", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_bandwidth.pdf"))
    plt.close(fig)
    g4 = E1.get("gray", "avx2", 1, 4096, 0, "gb_s")
    s4 = E1.get("sobel", "avx2", 1, 4096, 0, "gb_s")
    macro("gbGrayAvxFourK", fmt(g4, 1))
    macro("gbSobelAvxFourK", fmt(s4, 1))
    macro("pctGrayOfCopy", fmt(100 * g4 / cp[(1, 4096)], 0))
    macro("pctSobelOfCopy", fmt(100 * s4 / cp[(1, 4096)], 0))
    g42 = E1.get("gray", "avx2", 2, 4096, 0, "gb_s")
    macro("pctGrayOfCopyTwo", fmt(100 * g42 / cp[(2, 4096)], 0))
    macro("scaleGrayAvxFourK", fmt(E1.ms("gray", "avx2", 1, 4096) / E1.ms("gray", "avx2", 2, 4096), 2))
    macro("scaleConvAvxFourK", fmt(E1.ms("conv", "avx2", 1, 4096, 9) / E1.ms("conv", "avx2", 2, 4096, 9), 2))
    macro("scaleSobelAvxFourK", fmt(E1.ms("sobel", "avx2", 1, 4096) / E1.ms("sobel", "avx2", 2, 4096), 2))

    # ---- thread scaling figure
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labs, vals1, vals2 = [], [], []
    for op, k in (("gray", 0), ("sobel", 0), ("conv", 9)):
        for impl in ("scalar", "avx2"):
            labs.append(f"{label(op, k)}\n{impl}")
            vals1.append(E1.ms(op, impl, 1, 4096, k) / E1.ms(op, impl, 2, 4096, k))
    x = np.arange(len(labs))
    ax.bar(x, vals1, 0.55, color=[C["scalar"] if "scalar" in l else C["avx2"] for l in labs])
    ax.axhline(2, color="k", ls="--", lw=0.8)
    ax.text(len(labs) - 0.5, 2.03, "ideal (2 cores)", fontsize=7, ha="right")
    for xi, v in zip(x, vals1):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("speedup of 2 threads over 1")
    ax.set_ylim(0, 2.4)
    ax.set_title("OpenMP scaling on the 2-core sandbox, 4096x4096", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_threads.pdf"))
    plt.close(fig)

    # ---- E3: stride
    srows, pairs = [], []
    for w, strides in ((1024, (1024, 1088)), (2048, (2048, 2112)), (4096, (4096, 4160)), (8192, (8192, 8256))):
        d = {s: Data(load(os.path.join(res, f"E3_stride_w{w}_s{s}.csv"))) for s in strides}
        for op, impl, k in (("sobel", "scalar", 0), ("sobel", "avx2", 0), ("conv", "avx2", 9)):
            a = d[strides[0]].get(op, impl, 1, w, 9 if op == "conv" else 0)
            b = d[strides[1]].get(op, impl, 1, w, 9 if op == "conv" else 0)
            srows.append(f"{w} & {op} {impl}{' k=9' if op == 'conv' else ''} & {a:.2f} & {b:.2f} & {100 * (a / b - 1):+.1f}\\,\\% \\\\")
            pairs.append((w, op, impl, 100 * (a / b - 1)))
    for s in (4096, 4160):
        d = Data(load(os.path.join(res, f"E3b_stride_w4096_s{s}.csv")))
        for impl in ("scalar", "autovec", "avx2"):
            pass
    d0, d1 = (Data(load(os.path.join(res, f"E3b_stride_w4096_s{s}.csv"))) for s in (4096, 4160))
    for k in (15, 21):
        for impl in ("scalar", "autovec", "avx2"):
            a, b = d0.get("conv", impl, 1, 4096, k), d1.get("conv", impl, 1, 4096, k)
            srows.append(f"4096$\\times$512 & conv {impl} k={k} & {a:.2f} & {b:.2f} & {100 * (a / b - 1):+.1f}\\,\\% \\\\")
            pairs.append((4096, f"conv k={k}", impl, 100 * (a / b - 1)))
    write("tab_stride.tex", "\n".join(srows))
    vals = [p[3] for p in pairs]
    macro("strideWorstSlow", f"{max(vals):.0f}")
    macro("strideBestFast", f"{min(vals):.0f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(range(len(pairs)), vals, color=[C["avx2"] if p[2] == "avx2" else C["scalar"] if p[2] == "scalar" else C["autovec"] for p in pairs])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-max(spread1, 5), max(spread1, 5), color="gray", alpha=0.15)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{p[0]}\n{p[1].replace('sobel ', 'S').replace('conv ', 'C')}\n{p[2]}" for p in pairs], fontsize=4.8)
    ax.set_ylabel("extra time with stride = row size (%)")
    ax.set_title("Cost of a power-of-two stride versus a padded stride (positive = slower); grey band = run-to-run noise", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_stride.pdf"))
    plt.close(fig)

    # ---- OpenCV table
    ocv = os.path.join(res, "opencv_compare.csv")
    if os.path.exists(ocv):
        rows = []
        for r in load(ocv):
            name = {"gray": "gray", "sobel": "Sobel (3 OpenCV calls)", "conv": f"conv {r['ksize']}x{r['ksize']}"}[r["op"]]
            rows.append(f"{name} & {float(r['opencv_ms']):.2f} & {float(r['simdip_avx2_ms']):.2f} & {float(r['opencv_over_simdip']):.1f}$\\times$ \\\\")
        write("tab_opencv.tex", "\n".join(rows))
        o = {r["op"] + r["ksize"]: float(r["opencv_over_simdip"]) for r in load(ocv)}
        macro("ocvGray", fmt(o["gray0"], 1))
        macro("ocvSobel", fmt(o["sobel0"], 1))
        macro("ocvConvThree", fmt(o["conv3"], 1))
        macro("ocvConvNine", fmt(o["conv9"], 1))

    # ---- mutation table
    mut = os.path.join(ROOT, "results", "mutation_results.txt")
    if os.path.exists(mut):
        rows, killed, total = [], 0, 0
        for line in open(mut):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4 or parts[0] not in ("KILLED", "SURVIVED", "ERROR"):
                continue
            total += 1
            killed += parts[0] == "KILLED"
            how = "AddressSanitizer" if "AddressSanitizer" in parts[3] or "SUMMARY" in parts[3] else "tests"
            rows.append(f"{tex_escape(parts[2])} & {parts[0].lower()} ({how}) \\\\")
        write("tab_mutation.tex", "\n".join(rows))
        macro("mutKilled", killed)
        macro("mutTotal", total)

    # ---- vectorization report
    rep = os.path.join(ROOT, "build", "vec_report_full.txt")
    if os.path.exists(rep):
        text = open(rep).read()
        src = open(os.path.join(ROOT, "src", "scalar_impl.inc")).read().split("\n")
        asrc = open(os.path.join(ROOT, "src", "scalar_autovec.cpp")).read().split("\n")

        def line_of(lines, needle, after=None):
            start = 0
            if after:
                start = next(i for i, l in enumerate(lines) if after in l)
            return next(i for i in range(start, len(lines)) if needle in lines[i]) + 1

        entries = [("gray: loop over $x$ (3 interleaved channels)", "scalar_impl.inc", line_of(src, "for (int x = 0; x < w; ++x) {", "void rgbToGrayRows")),
                   ("Sobel: interior loop over $x$", "scalar_impl.inc", line_of(src, "for (int x = 1; x < w - 1; ++x)", "void sobelRows")),
                   ("convolution: loop over $x$ with the $k^2$ taps inside", "scalar_impl.inc", line_of(src, "for (int x = 0; x < w; ++x) {", "void convolveRows")),
                   ("convolution row-accumulate rewrite: \\code{a[x] += wv * s[x]}", "scalar\\_autovec.cpp", line_of(asrc, "a[x] += wv * s[x]"))]
        rows = []
        nvec = 0
        for desc, fname, ln in entries:
            fn = fname.replace("\\", "")
            vec = re.search(rf"{re.escape(fn)}:{ln}:\d+: optimized: loop vectorized using (\d+) byte", text)
            if vec:
                status, reason = f"\\textbf{{yes}} ({vec.group(1)}-byte vectors)", ""
                nvec += 1
            else:
                m = re.findall(rf"{re.escape(fn)}:(?:{ln}|{ln + 1}|{ln + 2}):\d+: missed: (?:not vectorized: )?([^\n]*)", text)
                m = [x for x in m if "couldn't vectorize loop" not in x]
                status = "no"
                reason = tex_escape(m[0].split(":")[0][:70]) if m else ""
            rows.append(f"{desc} & {fn.replace('_', chr(92) + '_')}:{ln} & {status} & {reason} \\\\")
        write("vec_report.tex", "\\begin{center}\\small\\begin{tabular}{@{}p{5.6cm}p{3.2cm}p{3.2cm}p{3.6cm}@{}}\\toprule\n"
              "\\textbf{loop} & \\textbf{source line} & \\textbf{vectorized by GCC?} & \\textbf{reason if not}\\\\ \\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")
        macro("gccVec", nvec)

    # ---- mask table (the deinterleave indices, from the same formula as the C++ constexpr)
    rows = []
    for c, nm in enumerate("RGB"):
        cells = []
        for s in range(3):
            idx = ["--"] * 16
            for i in range(16):
                g = 3 * i + c
                if g // 16 == s:
                    idx[i] = str(g % 16)
            cells.append(" ".join(f"{v:>2s}" for v in idx))
        rows.append(f"\\textbf{{{nm}}} & \\texttt{{{cells[0]}}} & \\texttt{{{cells[1]}}} & \\texttt{{{cells[2]}}} \\\\")
    write("mask_table.tex", "\\begin{tabular}{@{}cccc@{}}\\toprule\n & mask for $v_0$ & mask for $v_1$ & mask for $v_2$\\\\ \\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\n{\\footnotesize output byte $i$ = input byte at the listed index; \\texttt{--} means index \\texttt{0x80} (write 0)}")


def test_numbers(testlog, crosslog):
    if os.path.exists(testlog):
        t = open(testlog).read()
        m = re.search(r"(\d+) checks, (\d+) failures", t)
        macro("testChecks", f"{int(m.group(1)):,}".replace(",", "\\,"))
        macro("testFailures", m.group(2))
        macro("testGroups", len(re.findall(r"^\[ ok \]|^\[FAIL\]", t, re.M)))
        w = re.search(r"worst observed error / theoretical bound = ([\d.]+)", t)
        if w:
            macro("errBoundRatio", w.group(1))
    if os.path.exists(crosslog):
        t = open(crosslog).read()
        m = re.search(r"(\d+) comparisons, (\d+) mismatches", t)
        macro("crossComparisons", m.group(1))
        macro("crossMismatches", m.group(2))
        a = re.search(r"\(([\d.]+) %\), worst difference (\d+)", t)
        if a:
            macro("crossApproxPct", a.group(1))
            macro("crossApproxWorst", a.group(2))


def fig_scaling(res):
    """OpenMP scaling (cluster experiment E7): speedup over 1 thread versus thread count, against the memcpy scaling."""
    path = os.path.join(res, "E7_scaling.csv")
    if not os.path.exists(path):
        return
    rows = load(path)
    size = max(int(r["width"]) for r in rows)
    thr = sorted({int(r["threads"]) for r in rows})
    D = Data(rows)
    cases = [("gray", 0), ("sobel", 0), ("conv", 9)]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(thr, thr, "k:", lw=0.8, label="ideal")
    cpath = os.path.join(res, "E7_copy_scaling.csv")
    if os.path.exists(cpath):
        C7 = Data(load(cpath))
        try:
            ax.plot(thr, [C7.ms("copy", "memcpy", 1, size) / C7.ms("copy", "memcpy", t, size) for t in thr], "s--", ms=3.5, color="#b22222",
                    label="memcpy (bandwidth)")
        except KeyError:
            pass
    sty = {"gray": "o-", "sobel": "^-", "conv": "d-"}
    trows = []
    for op, k in cases:
        try:
            base = D.ms(op, "avx2", 1, size, k)
            ax.plot(thr, [base / D.ms(op, "avx2", t, size, k) for t in thr], sty[op], ms=3.5, color=C["avx2"] if op == "conv" else
                    ("#2e7d32" if op == "gray" else "#d98c1f"), label=f"AVX2 {label(op, k)}")
            trows.append(f"{label(op, k)} AVX2 & " + " & ".join(f"{D.ms(op, 'avx2', t, size, k):.2f}" for t in thr) + " \\\\")
            base_s = D.ms(op, "scalar", 1, size, k)
            trows.append(f"{label(op, k)} scalar & " + " & ".join(f"{D.ms(op, 'scalar', t, size, k):.2f}" for t in thr) + " \\\\")
        except KeyError:
            pass
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(thr)
    ax.set_xticklabels([str(t) for t in thr])
    ax.set_yticks(thr)
    ax.set_yticklabels([str(t) for t in thr])
    ax.set_xlabel("OpenMP threads (pinned, one per core)")
    ax.set_ylabel("speedup over 1 thread")
    ax.set_title(f"OpenMP scaling, {size}x{size} pixels", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f_scaling.pdf"))
    plt.close(fig)
    write("tab_scaling.tex", "\n".join(trows))
    write("tab_scaling_head.tex", " & ".join(str(t) for t in thr))


def main():
    global FIG, SNIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "sandbox_example"))
    ap.add_argument("--out", default=None, help="write figures/ and snippets/ below this directory instead of docs/")
    ap.add_argument("--testlog", default=os.path.join(ROOT, "results", "test_log.txt"))
    ap.add_argument("--crosslog", default=os.path.join(ROOT, "results", "crosscheck_log.txt"))
    args = ap.parse_args()
    if args.out:
        FIG, SNIP = os.path.join(args.out, "figures"), os.path.join(args.out, "snippets")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(SNIP, exist_ok=True)
    if not args.out:
        fig_gallery()
        fig_sobel_parts()
    results(args.results)
    fig_scaling(args.results)
    test_numbers(args.testlog, args.crosslog)
    with open(os.path.join(SNIP, "macros.tex"), "w") as f:
        for k, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(MACROS)} macros, figures and tables")


if __name__ == "__main__":
    main()
