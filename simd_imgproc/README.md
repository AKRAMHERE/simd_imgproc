# simd_imgproc: SIMD image-processing library (C++17, AVX2, OpenMP, Python)

Three image operations, each in three implementations and with optional OpenMP row-band parallelism:

| operation | definition |
|---|---|
| `rgbToGray` | `Y = (77 R + 150 G + 29 B + 128) >> 8` (integer) |
| `sobel` | `min(255, abs(Gx) + abs(Gy))` on a gray image (integer) |
| `convolve` | k x k float32 correlation, round-half-even, clamp to 0..255; k odd, 1..63 |

Implementations: `Impl::Scalar` (auto-vectorizer off), `Impl::AutoVec` (same source, vectorizer on), `Impl::Avx2` (intrinsics, run-time CPU check).
Border: replicate (clamp to edge). Images: 8-bit, row stride with 64-byte-aligned rows.

The full explanation (maths, code walkthrough, cache and numerics chapters, benchmark method, bwUniCluster guide,
interview material) is in `SIMD_Image_Processing_Guide.pdf`; its LaTeX sources are in `docs/`.

## Quick start

```bash
make -j4              # build/simdip (CLI), build/simdip_bench, build/simdip_tests, build/libsimdip.a
make test             # 24 test groups; last line must say "0 failures"
make asan-test        # same tests with AddressSanitizer + UBSan
python3 python/make_test_images.py
python3 python/crosscheck.py          # independent NumPy/SciPy reference (needs numpy, scipy)
bash scripts/mutation_test.sh > results/mutation_results.txt   # 12 seeded bugs must all be caught

./build/simdip sobel data/camera.pgm out.pgm --impl avx2 --threads 2
./build/simdip_bench --op gray,sobel,conv --impl scalar,autovec,avx2 --sizes 1024,2048 --ksizes 3,9 --threads 1,2 --csv out.csv
```

Requirements: GCC with C++17 and OpenMP; an x86-64 CPU with AVX2+FMA for the AVX2 path (without it the other paths still work).
`make OPENMP=0` builds without OpenMP. `make OPT="-O3 -march=native" BUILD=build_native` lets the compiler use the whole CPU in the autovec path.

## API

```cpp
#include "simdip/ops.hpp"
using namespace simdip;
Image rgb = readPNM("in.ppm");
Image gray(rgb.width(), rgb.height(), 1);
rgbToGray(rgb, gray, {Impl::Avx2, 4});           // implementation, OpenMP threads
Image blur(gray.width(), gray.height(), 1);
convolve(gray, blur, kernels::gaussian(9, 2.5), {Impl::Avx2, 4});
```

Contract: gray and Sobel are bit-identical in all implementations. AVX2 convolution equals `reference::convolveFmaFloat`
bit for bit; scalar/autovec convolution may differ from it by one grey level on rare near-ties (see guide, chapter 11).
Wrong sizes, overlapping buffers or unusable kernels throw `std::invalid_argument`.

## On bwUniCluster 3.0

```bash
rsync -av --exclude 'build*' --exclude results/sandbox_example simd_imgproc/ <user>@uc3.scc.kit.edu:~/simd_imgproc/
cd ~/simd_imgproc && mkdir -p logs results
sbatch slurm/01_build_test.sbatch      # build, tests, ASAN tests, NumPy cross-check (dev_cpu)
bash slurm/submit_all.sh               # 02 benchmarks (2 cores) and 03 OpenMP scaling (32 cores) on cpu_il and cpu
sbatch slurm/04_profile.sbatch         # vectorization report + gprof (dev_cpu)
```

Results land in `results/<partition>/`. Turn them into the guide's figures/tables with
`python3 python/make_figures.py --results results/cpu_il --out results/cpu_il/report`.
Details, queue limits and troubleshooting: guide chapter 15.

## What is and is not measured

`results/sandbox_example/` holds timings from a 2-core virtual machine in the sandbox where the code was written. They show how
the experiments look; they are **not** bwUniCluster numbers. `results/sandbox_example/firstrun/` keeps an earlier pass of the same
experiments (the scalar baseline differed between passes, see guide chapter 13). `results/test_log.txt`,
`crosscheck_log.txt` and `mutation_results.txt` are the logs of the corresponding runs.

## Layout

`include/simdip/` API · `src/` implementations · `apps/` CLI and benchmark · `tests/` test suite · `python/` reference,
cross-check, experiments, figures · `scripts/` mutation test · `slurm/` cluster jobs · `docs/` guide sources · `data/` generated test images.

## Known limits

8-bit images only; one border mode; square odd kernels without separable-kernel shortcut; no AVX-512 path; no row blocking in
the convolution (the next optimization step); thread-count tests show identical output, they are not a formal race-freedom proof.
