#!/usr/bin/env python3
"""Runs the full experiment set locally by calling build/simdip_bench, one CSV per experiment.

  python3 python/run_experiments.py --outdir results/sandbox_example [--quick]

Experiments (E1..E6 are described in the guide, chapter "Measuring performance"):
  E1 main      all ops x impls x sizes x kernel sizes x threads
  E2 sizes     throughput versus image size (caches)
  E3 stride    row stride = exact power of two versus padded
  E4 ksize     convolution versus kernel size, including the ablation variants
  E5 copy      memory-copy bandwidth (the yardstick for the memory-bound kernels)
The numbers depend on the machine; the machine is recorded in meta.txt next to the CSVs.
"""
import argparse
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "build", "simdip_bench")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def run(args, csv, log):
    cmd = [BENCH] + args + ["--csv", csv]
    t0 = time.time()
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"  {os.path.basename(csv):28s} {time.time() - t0:6.1f} s  rc={r.returncode}", flush=True)
    if r.returncode != 0:
        sys.exit("benchmark failed, see " + log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "sandbox_example"))
    ap.add_argument("--quick", action="store_true", help="tiny version to check that everything runs")
    ap.add_argument("--threads", default="1,2", help="thread counts for E1")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    log = os.path.join(a.outdir, "log.txt")
    open(log, "w").close()
    with open(os.path.join(a.outdir, "meta.txt"), "w") as f:
        f.write(f"date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}\nhost: {platform.node()}\n")
        f.write(f"cpu: {sh('grep -m1 \"model name\" /proc/cpuinfo | cut -d: -f2').strip()}\n")
        f.write(f"cores visible: {os.cpu_count()}\n")
        f.write(f"lscpu caches:\n{sh('lscpu | grep -E \"L1d|L1i|L2|L3\"')}\n")
        f.write(f"compiler: {sh('g++ --version | head -1')}\n")
        f.write(f"kernel: {platform.release()}\n")
    q = a.quick
    reps = "3" if q else str(a.reps)
    sizes = "256,512" if q else "256,512,1024,2048,4096"
    ks = "3,5" if q else "3,5,7,9,11"
    print("E1 main"); run(["--op", "gray,sobel,conv", "--impl", "scalar,autovec,avx2", "--sizes", sizes, "--ksizes", ks,
                         "--threads", a.threads, "--reps", reps, "--label", "E1"], os.path.join(a.outdir, "E1_main.csv"), log)
    print("E2 size sweep"); run(["--op", "gray,sobel,conv", "--impl", "scalar,avx2", "--sizes",
                                 "64,128,256,512,1024,2048,4096" if q else "64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192",
                                 "--ksizes", "5", "--threads", "1", "--reps", reps, "--min-ms", "10", "--label", "E2"],
                                os.path.join(a.outdir, "E2_sizes.csv"), log)
    print("E3 stride")
    for w, strides in ((4096, (4096, 4160, 4224, 4352)), (2048, (2048, 2112)), (1024, (1024, 1088)), (8192, (8192, 8256))):
        if q and w > 2048:
            continue
        for s in strides:
            run(["--op", "sobel,conv", "--impl", "scalar,avx2", "--sizes", str(w), "--ksizes", "9", "--threads", "1",
                 "--reps", reps, "--stride", str(s), "--label", f"E3_stride{s}"], os.path.join(a.outdir, f"E3_stride_w{w}_s{s}.csv"), log)
    print("E4 kernel size"); run(["--op", "conv", "--impl", "scalar,autovec,avx2", "--sizes", "512" if q else "1024",
                                  "--ksizes", "3,5" if q else "3,5,7,9,11,15,21,31", "--threads", "1", "--reps", reps,
                                  "--ablation", "--label", "E4"], os.path.join(a.outdir, "E4_ksize.csv"), log)
    print("E5 copy bandwidth"); run(["--op", "copy", "--sizes", "1024,2048" if q else "512,1024,2048,4096,8192", "--threads", a.threads,
                                     "--reps", reps, "--min-ms", "20", "--label", "E5"], os.path.join(a.outdir, "E5_copy.csv"), log)
    print("done ->", a.outdir)


if __name__ == "__main__":
    main()
