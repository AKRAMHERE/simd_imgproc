#!/bin/bash
# What did the compiler vectorize?  And a function-level view of where time goes (gprof).
#   sbatch slurm/04_profile.sbatch
#SBATCH --job-name=simdip-prof
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6000mb
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
mkdir -p results

make vec-report | tail -1 || true
cp build/vec_report_full.txt results/vec_report_cluster.txt
echo "vectorization report: results/vec_report_cluster.txt  ($(wc -l < results/vec_report_cluster.txt) lines)"

make BUILD=build_prof OPT="-O2 -pg -fno-inline" LDFLAGS="-pg -fopenmp" -j 2
WORK="$(mktemp -d)"
( cd "${WORK}" && OMP_NUM_THREADS=1 "${SLURM_SUBMIT_DIR}/build_prof/simdip_bench" --op sobel,conv --impl scalar,autovec \
        --sizes 1024 --ksizes 5 --threads 1 --reps 3 --min-ms 5 > /dev/null )
gprof -b -p build_prof/simdip_bench "${WORK}/gmon.out" > results/gprof_flat.txt
head -15 results/gprof_flat.txt
