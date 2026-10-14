#!/bin/bash
# Build the library on a compute node, run the unit tests, then the sanitizer build of the tests,
# the NumPy cross-check (if python3 + numpy are available) and the vectorization report.
# Uses the short development queue (30 min limit, one running job at a time).
#   sbatch slurm/01_build_test.sbatch
#SBATCH --job-name=simdip-build
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6000mb
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
mkdir -p results

make clean
make -j "${SLURM_CPUS_PER_TASK:-1}"
./build/simdip_tests | tee results/test_log_cluster.txt
tail -1 results/test_log_cluster.txt

# AddressSanitizer + UBSan build of the same tests (catches vector loads/stores outside the row)
make clean
make -j "${SLURM_CPUS_PER_TASK:-1}" asan-test 2>&1 | tail -3

# Independent NumPy reference. Skipped (not failed) if numpy is not available in the default python3.
if python3 -c "import numpy" 2>/dev/null; then
    make -j "${SLURM_CPUS_PER_TASK:-1}"
    python3 python/make_test_images.py 2>/dev/null || true
    python3 python/crosscheck.py | tee results/crosscheck_log_cluster.txt | tail -3
else
    echo "numpy not found for python3: skipping python/crosscheck.py (module load devel/python ... if you want it)"
fi
echo "build + tests OK on $(hostname)"
