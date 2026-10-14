#!/bin/bash
# Benchmarks on TWO cores of the partition you submit to (single-thread numbers, plus 1 vs 2 threads):
# scalar vs autovec vs AVX2, sizes, kernel sizes, ablations, stride experiment, memcpy yardstick, repeatability.
#   sbatch --partition=cpu_il slurm/02_bench_single.sbatch     # Intel Xeon Platinum 8358 (Ice Lake)
#   sbatch --partition=cpu    slurm/02_bench_single.sbatch     # AMD EPYC 9454
#   NATIVE=1 sbatch --partition=cpu_il slurm/02_bench_single.sbatch   # also rebuild with -march=native
# Add --exclusive to the sbatch line for the cleanest numbers (you then wait longer in the queue).
#SBATCH --job-name=simdip-b1
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000mb
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

PART="${SLURM_JOB_PARTITION:-local}"
BIN=build
TAG="${PART}"
if [[ "${NATIVE:-0}" == "1" ]]; then
    BIN="build_native_${PART}"
    TAG="${PART}_native"
    make -j 2 BUILD="${BIN}" OPT="-O3 -march=native"
elif [[ ! -x build/simdip_bench ]]; then
    make -j 2
fi
B="${BIN}/simdip_bench"
R="results/${TAG}"
mkdir -p "${R}"

# --- record exactly what the numbers were measured on
{
    echo "date:       $(date -Is)"
    echo "host:       $(hostname)"
    echo "partition:  ${PART}"
    echo "job id:     ${SLURM_JOB_ID:-none}"
    echo "compiler:   $(g++ --version | head -1)"
    echo "binary dir: ${BIN}"
    echo "git/tar:    $(cat VERSION 2>/dev/null || echo unknown)"
    echo "cpu:        $(lscpu | grep -m1 'Model name' | sed 's/.*: *//')"
    echo "has avx2/fma: $(grep -o -w -m2 -E 'avx2|fma' /proc/cpuinfo | sort -u | tr '\n' ' ')"
    lscpu | grep -E 'Socket|Thread|Core|L1d|L2|L3' || true
} > "${R}/meta.txt"
cat "${R}/meta.txt"

# The experiment set below is exactly the one behind the sandbox example in results/sandbox_example/, so that
#   python3 python/make_figures.py --results results/<partition> --out results/<partition>/report
# can turn it into the same figures and tables as in the guide. Thread counts 1 and 2 only: this job owns 2 cores.
export OMP_PROC_BIND=close OMP_PLACES=cores
RUN() { srun --cpu-bind=cores --cpus-per-task=2 "${B}" "$@"; }

# E1  every operation x implementation x size x kernel size, 1 and 2 threads
RUN --op gray,sobel,conv --impl scalar,autovec,avx2 --sizes 256,512,1024,2048,4096 --ksizes 3,5,7,9,11 \
    --threads 1,2 --reps 7 --label E1 --csv "${R}/E1_main.csv" > "${R}/E1_main.txt"

# E2  throughput versus image size (cache effects), 1 thread
RUN --op gray,sobel,conv --impl scalar,avx2 --ksizes 5 --threads 1 --reps 7 --min-ms 10 --label E2 \
    --sizes 64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192 --csv "${R}/E2_sizes.csv" > "${R}/E2_sizes.txt"

# E3  row stride: exactly width (page multiple for 4096 and 8192) versus width + 64 bytes
for W in 1024 2048 4096 8192; do
    for S in "${W}" $((W+64)); do
        RUN --op sobel,conv --impl scalar,avx2 --sizes "${W}" --ksizes 9 --threads 1 --stride "${S}" --reps 7 \
            --label "E3_w${W}_s${S}" --csv "${R}/E3_stride_w${W}_s${S}.csv" > /dev/null
    done
done
# E3b  wide, flat image (4096 x 512) with large kernels: the working set is mostly the float ring
for S in 4096 4160; do
    RUN --op conv --impl scalar,autovec,avx2 --sizes 4096 --aspect 8 --ksizes 15,21 --threads 1 --stride "${S}" --reps 7 \
        --label "E3b_s${S}" --csv "${R}/E3b_stride_w4096_s${S}.csv" > /dev/null
done

# E4  convolution versus kernel size on 1024 x 1024 with the three ablations
RUN --op conv --impl scalar,autovec,avx2 --ablation --sizes 1024 --ksizes 3,5,7,9,11,15,21,31 --threads 1 --reps 7 \
    --label E4 --csv "${R}/E4_ksize.csv" > "${R}/E4_ksize.txt"

# E5  memory-bandwidth yardstick (memcpy of the same pixels)
RUN --op copy --sizes 512,1024,2048,4096,8192 --threads 1,2 --reps 7 --label E5 --csv "${R}/E5_copy.csv" > "${R}/E5_copy.txt"

# E6  repeatability: the same measurement three times in one job
for I in 1 2 3; do
    RUN --op gray,sobel,conv --impl scalar,autovec,avx2 --sizes 2048 --ksizes 3,9 --threads 1,2 --reps 7 \
        --label "E6_${I}" --csv "${R}/E6_repeat_${I}.csv" > /dev/null
done

# OpenCV comparison (optional, only if python3 has cv2): single-thread AVX2 numbers versus OpenCV
if python3 -c "import cv2, numpy" 2>/dev/null; then
    RUN --op gray,sobel,conv --impl avx2 --sizes 2048 --ksizes 3,5,9 --threads 1 --reps 9 --label cmp --csv "${R}/_cmp.csv" > /dev/null
    python3 python/compare_opencv.py --bench-csv "${R}/_cmp.csv" --size 2048 --out "${R}/opencv_compare.csv" | tail -8 || true
fi
echo "done: ${R}/"
